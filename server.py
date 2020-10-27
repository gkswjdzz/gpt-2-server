from flask import Flask, request, abort, jsonify
import requests
from lib import encode, decode
import json

app = Flask(__name__)

SERVER='https://length-1-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict'

@app.route('/large', methods=['POST'])
def large():
  keys = list(request.form.keys())

  if len(keys) != 1 :
    return jsonify({'message': 'invalid body'}), 400
  
  raw_text_key = list(request.form.keys())[0]
  print(raw_text_key)
  raw_text = request.form[raw_text_key]
  print(encode(raw_text))
  
  data = {
    'signature_name':'predict',
    'instances':[
      encode(raw_text)
    ]
  }
  res = requests.post(SERVER, data=json.dumps(data))
  
  if res.status_code is not 200:
    return jsonify({'error'}), res.status_code
  r = res.json()
  return decode(r['predictions']), 200


if __name__ == "__main__":
  app.run(debug=False, port=8000, host='0.0.0.0', threaded=False)