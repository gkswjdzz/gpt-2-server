from flask import Flask, request, abort, jsonify
import requests
from lib import encode, decode
import json

app = Flask(__name__)

SERVERS = {
  'length-1': 'https://length-1-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict',
  'length-x': 'https://main-gpt-2-large-tf-serving-gkswjdzz.endpoint.ainize.ai/v1/models/gpt-2-large:predict'
}
@app.route('/large', methods=['POST'])
def large():
  keys = list(request.form.keys())

  if len(keys) != 2 :
    return jsonify({'message': 'invalid body'}), 400
  
  raw_text_key = list(request.form.keys())[0]
  length_key = list(request.form.keys())[1] 

  print(raw_text_key)
  raw_text = request.form[raw_text_key]
  
  length = request.form[length_key]
  
  print(('length-'+length))
  if not(('length-'+ length) in SERVERS.keys()):
    return jsonify({'message': 'body error'}), 400
  
  
  print(encode(raw_text))
  print(length, type(length))

  
  data = {
    'signature_name':'predict',
    'instances':[
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
