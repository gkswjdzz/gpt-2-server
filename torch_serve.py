import requests
import os


def get_scale_model(model_name):
    path= f'/models/{model_name}'
    post_path = MANAGEMENT_URL + path
    res = requests.get(post_path)

    if res.status_code != 200:
        return 0

    res = res.json()
    print(res)
    print(res[0]['minWorkers'])
    worker = res[0]['minWorkers']
    return worker


def set_scale_model(model_name, scale):
    path= f'/models/{model_name}'
    params= {
        'min_worker': scale,
        'max_worker': scale,
        'synchronous': True
    }
    post_path = MANAGEMENT_URL + path
    res = requests.post(post_path, params=params)

    if res.status_code != 200:
        return None

    res = res.json()
    print(res)
    return res


def register_model(model_name):
    path= '/models'
    params= {
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
