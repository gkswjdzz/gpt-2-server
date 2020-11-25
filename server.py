from flask import Flask, request, jsonify
import requests
from lib import encode, decode
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

app = Flask(__name__)

SERVERS = {
  'length-1': 'https://length-1-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict',
  'length-x': 'https://main-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict'
}

TORCH_MODELS = {
  'base': 'https://base-gpt-2-large-torch-serving-gkswjdzz.endpoint.ainize.ai/predictions/gpt2-large'
}


@app.route('/preprocess', methods=['POST'])
def preprocess():
    if request.is_json:
        content = request.get_json()
        return jsonify(json.dumps(tokenizer(content['context'])['input_ids'])), 200
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

    raw_text_key = list(request.form.keys())[0]
    num_samples_key = list(request.form.keys())[1]
    length_key = list(request.form.keys())[2]

    raw_text = request.form[raw_text_key]
    num_samples = request.form[num_samples_key]
    length = request.form[length_key]

    encoded_text = tokenizer.encode(raw_text)

    data = {
        'text': encoded_text,
        'num_samples': int(num_samples),
        'length': int(length)
    }

    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post('https://base-gpt-2-large-torch-serving-gkswjdzz.endpoint.ainize.ai/predictions/gpt2-large', headers=headers, data=json.dumps(data))

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code

    response = res.json()

    result = dict()
    for idx, sample_output in enumerate(response):
        result[idx] = tokenizer.decode(sample_output, skip_special_tokens=True)

    return result, 200


@app.route('/large', methods=['POST'])
def large():
    keys = list(request.form.keys())

    if len(keys) != 2:
        return jsonify({'message': 'invalid body'}), 400

    raw_text_key = list(request.form.keys())[0]
    length_key = list(request.form.keys())[1]

    raw_text = request.form[raw_text_key]

    length = request.form[length_key]

    if not(('length-' + length) in SERVERS.keys()):
        return jsonify({'message': 'body error'}), 400

    data = {
        'signature_name': 'predict',
        'instances': [
            encode(raw_text)
        ]
    }

    res = requests.post(SERVERS['length-'+length], data=json.dumps(data))

    if res.status_code != 200:
        return jsonify({'message': 'error'}), res.status_code
    r = res.json()
    return decode(r['predictions']), 200


if __name__ == "__main__":
    app.run(debug=False, port=8000, host='0.0.0.0', threaded=False)
