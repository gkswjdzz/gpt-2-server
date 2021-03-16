import requests
import os
import json
import subprocess
import threading
import time

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

lock = threading.Lock()

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

def set_scale_0_least_recently_used():
    f = open('db.txt', 'r')
    lines = f.readlines()
    f.close()

    if lines:
        model_name, _ = lines[0].split(',')
        if set_scale_model(model_name, 0) == None:
            return False
        print(f'stop {model_name}') 

    f = open('db.txt','w')
    for line in lines[1:]:
        key, value = line.split(',')
        f.write(f'{key},{value}')
    f.close()

    return True

def set_scale_model(model_name, scale, sleep=0):
    time.sleep(sleep)
    if scale > 0:
        print('before lock1...')
        print('after lock1...')
        out = subprocess.check_output('nvidia-smi --query-gpu="memory.free" --format=csv,noheader', shell=True).decode('utf-8')
        output = out.split('\n')
        output = [out.split(' ')[0] for out in output]
        output = list(filter(lambda x: x, output))
        gpu_resource = [int(x) for x in output]
        print('end out1')
        out = subprocess.check_output(os.path.join(f'du {MODEL_STORE_PATH}/', f'{model_name}.mar'), shell=True).decode('utf-8')
        print(model_name)
        out = out.split()
        if len(out) == 0:
            return None
    
        model_size = int(out[0]) / 1000

        print('model_size : {} MB'.format(model_size))
        print('maximum free memory of gpus: {} MB'.format(max(gpu_resource)))

        # MB
        if model_size < 500 and max(gpu_resource) < 2000:
            if not set_scale_0_least_recently_used():
                return jsonify({'message':'Fail to scale 0'}), 500
            return set_scale_model(model_name, scale, 1)
        elif 500 <= model_size < 2000 and max(gpu_resource) < 2500:
            if not set_scale_0_least_recently_used():
                return jsonify({'message':'Fail to scale 0'}), 500
            return set_scale_model(model_name, scale, 1)
        elif 2000 <= model_size < 3000 and max(gpu_resource) < 4500:
            if not set_scale_0_least_recently_used():
                return jsonify({'message':'Fail to scale 0'}), 500
            return set_scale_model(model_name, scale, 1)

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
