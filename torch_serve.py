import requests
import os
import json

TORCH_MODELS = {
    'base': os.environ.get('GPT2_LARGE_BASE_TORCH_SERVER'),
    'gpt3': os.environ.get('GPT3_BASED_GPT2_TORCH_SERVER')
}


# inference 8080
INFERENCE_URL = 'http://localhost:8080'
# management 8081
MANAGEMENT_URL = 'http://localhost:8081'
# inbound port 8000
# model-store directory
MODEL_STORE_PATH = './model-store'


def get_scale_model(model_name):
    path = f'/models/{model_name}'
    post_path = MANAGEMENT_URL + path
    res = requests.get(post_path)

    if res.status_code != 200:
        return None

    res = res.json()
    print(res)
    print(res[0]['minWorkers'])
    worker = res[0]['minWorkers']
    return worker


def set_scale_model(model_name, scale):
    path = f'/models/{model_name}'
    params = {
        'min_worker': scale,
        'max_worker': scale,
        'synchronous': True
    }
    post_path = MANAGEMENT_URL + path
    res = requests.put(post_path, params=params)

    if res.status_code != 200:
        return None

    res = res.json()
    print(res)
    return res


def register_model(model_name):
    path = '/models'
    params = {
        'url': os.path.join(MODEL_STORE_PATH, f'{model_name}.mar'),
        'synchronous': True
    }
    post_path = MANAGEMENT_URL + path
    res = requests.post(post_path, params=params)

    if res.status_code != 200:
        return None

    res = res.json()
    print(res)
    return res


def inference_model(model, data):
    path = f'/predictions/{model}'
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post(
        INFERENCE_URL + path, headers=headers, data=json.dumps(data)
    )

    if res.status_code != 200:
        return None

    return res.json()
