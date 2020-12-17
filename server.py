from flask import Flask, request, jsonify
import requests
from lib import encode, decode
import json
from transformers import AutoTokenizer
import emoji

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

app = Flask(__name__)

SERVERS = {
  'length-1': 'https://length-1-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict',
  'length-x': 'https://main-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict'
}

TORCH_MODELS = {
  'base': 'https://base-gpt-2-large-torch-serving-gkswjdzz.endpoint.ainize.ai/predictions/gpt2-large'
}

def removeEmoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def translateString(inputText):
    transDict = {
        '‘': '\'',
        '’': '\'',
        '“': '\"',
        '”': '\"',
        '\u2013': '-',
        '\u2014': '-',
        '\u3000': ' ',
    }

    return inputText.translate(str.maketrans(transDict))


@app.route('/preprocess', methods=['POST'])
def preprocess():
    if request.is_json:
        content = request.get_json()
        replacedContent = translateString(content['context'])
        replacedContent = removeEmoji(replacedContent)
        return jsonify(json.dumps(tokenizer(replacedContent)['input_ids'])), 200
    return jsonify(json.dumps([-1])), 400


@app.route('/postprocess', methods=['POST'])
def postprocess():
    if request.is_json:
        contents = request.get_json()
        result = {}
        for idx, content in enumerate(contents):
            result[idx] = {'text': tokenizer.decode(content)}
        return jsonify(result), 200
    return jsonify({}), 400


@app.route('/torch-serve', methods=['POST'])
def torch():
    keys = list(request.form.keys())

    if len(keys) != 3:
        return jsonify({'message': 'invalid body'}), 400

    rawTextKey = list(request.form.keys())[0]
    numResultsRequestKey = list(request.form.keys())[1]
    lengthKey = list(request.form.keys())[2]

    rawText = request.form[rawTextKey]
    numResultsRequest = request.form[numResultsRequestKey]
    length = request.form[lengthKey]

    encodedText = tokenizer.encode(rawText)

    data = {
        'text': encodedText,
        'num_samples': int(numResultsRequest),
        'length': int(length)
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post('https://base-gpt-2-large-torch-serving-gkswjdzz.endpoint.ainize.ai/predictions/gpt2-large', headers=headers, data=json.dumps(data))

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code

    response = res.json()

    result = dict()
    for idx, sampleOutput in enumerate(response):
        result[idx] = tokenizer.decode(sampleOutput, skip_special_tokens=True)

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
    app.run(debug=False, port=8000, host='0.0.0.0', threaded=False)
