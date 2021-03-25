import sentry_sdk
from flask import Flask, request, jsonify
from sentry_sdk.integrations.flask import FlaskIntegration
import os
import requests
from lib import encode, decode
import json
from transformers import AutoTokenizer, BertTokenizerFast
import emoji
from time import time

from cronjob import start_job, write_info
from torch_serve \
    import register_model, get_scale_model, set_scale_model, inference_model

env = os.environ.get('PRODUCT_ENV')

if env == "production":
    sentry_sdk.init(
        dsn=os.environ.get('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0
    )

GPT3_MODEL_NAME = "kykim/gpt3-kor-small_based_on_gpt2"
autoTokenizer = AutoTokenizer.from_pretrained("gpt2-large")
# From https://github.com/kiyoungkim1/LMkor
gpt3Tokenizer = BertTokenizerFast.from_pretrained(GPT3_MODEL_NAME)

app = Flask(__name__)

SERVERS = {
  'length-1': os.environ.get('GPT2_LARGE_LENGTH_1_TENSORFLOW_SERVING_URL'),
  'length-x': os.environ.get('GPT2_LARGE_LENGTH_X_TENSORFLOW_SERVING_URL')
}

TORCH_MODELS = {
    'base': os.environ.get('GPT2_LARGE_BASE_TORCH_SERVER'),
    'gpt3': os.environ.get('GPT3_BASED_GPT2_TORCH_SERVER')
}


def send_message_to_slack(text):
    url = os.environ.get('SLACK_INCOMING_WEBHOOKS_URL')
    payload = {
        "pretext": "*GPT2-AINIZE-API SERVER ERROR OCCURED!*",
        "text": f"*ERROR*: {text}",
        "color": "danger",
    }
    requests.post(url, json=payload)

    if env == "production":
        sentry_sdk.capture_message(text, "fatal")


def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def translate_string(input_text):
    trans_dict = {
        '‘': '\'',
        '’': '\'',
        '“': '\"',
        '”': '\"',
        '\u2013': '-',
        '\u2014': '-',
        '\u3000': ' ',
    }

    return input_text.translate(str.maketrans(trans_dict))


@app.route("/healthz", methods=["GET"])
def health_check():
    return "OK", 200


@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if request.is_json:
            content = request.get_json()
            sliced_content = content['context'][:1024]
            if len(sliced_content) == 0:
                return jsonify(json.dumps([-1])), 400
            replaced_content = translate_string(sliced_content)
            replaced_content = remove_emoji(replaced_content)
            vector = autoTokenizer(
                replaced_content, max_length=1024, truncation=True
            )['input_ids']
            return jsonify(json.dumps(vector)), 200
        return jsonify(json.dumps([-1])), 400
    except Exception as e:
        if request.is_json:
            send_message_to_slack(
                '*PRE_PROCESS*' +
                f'\n requested json: *{request.get_json()}*. \n *{e}*')
        else:
            send_message_to_slack(
                '*PRE_PROCESS*' +
                f'\n requested data: *{request.data}*. \n *{e}*')
        return jsonify(json.dumps([-1])), 500


@app.route('/postprocess', methods=['POST'])
def postprocess():
    try:
        if request.is_json:
            contents = request.get_json()
            result = {}
            for idx, content in enumerate(contents):
                if len(content) != 0:
                    content.pop()
                result[idx] = {'text': autoTokenizer.decode(content)}
            return jsonify(result), 200
        return jsonify({}), 400
    except Exception as e:
        if request.is_json:
            send_message_to_slack(
                '*POST_PROCESS*' +
                f'\n requested json: *{request.get_json()}*. \n *{e}*')
        else:
            send_message_to_slack(
                '*POST_PROCESS*' +
                f'\n requested data: *{request.data}*. \n *{e}*')
        return jsonify({}), 500


@app.route('/torch-serve', methods=['POST'])
def torch_serve():
    keys = list(request.form.keys())

    if len(keys) != 3:
        return jsonify({'message': 'invalid body'}), 400

    rawTextKey = list(request.form.keys())[0]
    numResultsRequestKey = list(request.form.keys())[1]
    lengthKey = list(request.form.keys())[2]

    rawText = request.form[rawTextKey]
    numResultsRequest = request.form[numResultsRequestKey]
    length = request.form[lengthKey]

    encodedText = autoTokenizer.encode(rawText)

    data = {
        'text': encodedText,
        'num_samples': int(numResultsRequest),
        'length': int(length)
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post(
        TORCH_MODELS['base'], headers=headers, data=json.dumps(data))

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code

    response = res.json()

    result = dict()
    for idx, sampleOutput in enumerate(response):
        result[idx] = autoTokenizer.decode(
            sampleOutput, skip_special_tokens=True)

    return result, 200


@app.route('/infer/<model>', methods=['POST'])
def torch_serve_inference(model):
    data = request.get_json()

    print(f'get data: {data}')

    if not data:
        return 'error'

    if model == 'gpt3':
        data['text'] = gpt3Tokenizer.encode(data['text'])
    else:
        data['text'] = autoTokenizer(data['text'])['input_ids']

    print(f'encoded data: {data["text"]}')

    scale = get_scale_model(model)

    print(f'get scale: {scale}')

    # model is not registered or scale = 0
    if not scale:
        if scale is None:
            ret = register_model(model)
            if ret is None:
                return jsonify({'message': 'model not found!'})
        ret = set_scale_model(model, 1)
        if ret is None:
            return jsonify({'message': 'model not  found!'}), 200
        elif ret == -1:
            return jsonify({'message': 'Too many request! Please try again in a little while.'}), 429
        print('set scale to 1')

    response = inference_model(model, data)
    print(f'result of inference: {response}')

    write_info(model, int(time()))
    print(f'write info of {model}')

    # if None
    result = dict()
    for idx, sampleOutput in enumerate(response):
        if model == 'gpt3':
            result[idx] = gpt3Tokenizer.decode(
                sampleOutput, skip_special_tokens=True)
        else:
            result[idx] = autoTokenizer.decode(
                sampleOutput, skip_special_tokens=True)

    return result, 200


@app.route('/infer/torch-gpt3-kor', methods=['POST'])
def torch_gpt3():
    keys = list(request.form.keys())

    if len(keys) != 3:
        return jsonify({'message': 'invalid body'}), 400

    rawTextKey = list(request.form.keys())[0]
    numResultsRequestKey = list(request.form.keys())[1]
    lengthKey = list(request.form.keys())[2]

    rawText = request.form[rawTextKey]
    numResultsRequest = request.form[numResultsRequestKey]
    length = request.form[lengthKey]

    encodedText = gpt3Tokenizer.encode(rawText)
    data = {
        'text': encodedText,
        'num_samples': int(numResultsRequest),
        'length': int(length)
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post(
        TORCH_MODELS['gpt3'], headers=headers, data=json.dumps(data)
    )

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code

    response = res.json()

    result = dict()
    for idx, sampleOutput in enumerate(response):
        result[idx] = gpt3Tokenizer.decode(
            sampleOutput, skip_special_tokens=True)

    return result, 200


@app.route('/large', methods=['POST'])
def large():
    keys = list(request.form.keys())

    if len(keys) != 2:
        return jsonify({'message': 'invalid body'}), 400

    rawTextKey = list(request.form.keys())[0]
    lengthKey = list(request.form.keys())[1]

    rawText = request.form[rawTextKey]

    length = request.form[lengthKey]

    if not(('length-' + length) in SERVERS.keys()):
        return jsonify({'message': 'body error'}), 400

    data = {
        'signature_name': 'predict',
        'instances': [
            encode(rawText)
        ]
    }

    res = requests.post(SERVERS['length-'+length], data=json.dumps(data))

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code
    r = res.json()
    return decode(r['predictions']), 200


if __name__ == "__main__":
    print('start job')
    start_job()
    print('start server')
    app.run(debug=True, port=8000, host='0.0.0.0', threaded=True)
