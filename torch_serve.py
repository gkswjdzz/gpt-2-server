import requests
import os
import json
import subprocess
import threading
import time
import torch

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

global VERY_BUSY
VERY_BUSY = False

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


def get_all_gpu_usages():
    path = "/models"
    get_path = MANAGEMENT_URL + path
    res = requests.get(get_path)
    device_count = torch.cuda.device_count()
    result = {}
    for i in range(device_count):
        result[i] = []

    if res.status_code != 200:
        return -1

    models = res.json()["models"]

    for model in models:
        model_name = model["modelName"]
        gpu_id = get_gpu_usage(model_name)
        print(gpu_id)
        if gpu_id != -1:
            result[gpu_id].append(model_name)
        
    return result


def get_gpu_usage(model_name):
    path = f"/models/{model_name}"
    get_path = MANAGEMENT_URL + path
    res = requests.get(get_path)
    if res.status_code != 200:
        return -1

    models_status = res.json()
    num_workers = len(models_status)
    if num_workers == 0:
        return -1

    model_status = models_status[0]

    if len(model_status["workers"]) == 0:
        return -1

    usage = model_status["workers"][0]["gpuUsage"]
    gpu_id = int(usage.split("::")[1].split(' ')[0])
    return gpu_id


def set_scale_0_least_recently_used(gpu_id):
    f = open('db.txt', 'r')
    lines = f.readlines()
    f.close()
    removed_model_name = None
    models = get_all_gpu_usages()[gpu_id]
    if lines:
        for idx, line in enumerate(lines):
            model_name, _ = line.split(',')
            if model_name in models:
                removed_model_name = model_name
                if set_scale_model(model_name, 0) is None:
                    return False
                print(f'stop {model_name}')
                break

    f = open('db.txt', 'w')
    for line in lines:
        key, value = line.split(',')
        if key != removed_model_name:
            f.write(f'{key},{value}')
    f.close()

    return True


def set_scale_model(model_name, scale, sleep=0):
    global VERY_BUSY
    time.sleep(sleep)
    if scale > 0:
        out = subprocess.check_output('nvidia-smi --query-gpu="memory.free" --format=csv,noheader', shell=True).decode('utf-8')
        output = out.split('\n')
        output = [out.split(' ')[0] for out in output]
        output = list(filter(lambda x: x, output))
        gpu_resource = [int(x) for x in output]
        out = subprocess.check_output(os.path.join(f'du {MODEL_STORE_PATH}/', f'{model_name}.mar'), shell=True).decode('utf-8')
        out = out.split()
        if len(out) == 0:
            return None
    
        model_size = int(out[0]) / 1000

        print('model_size : {} MB'.format(model_size))
        print('minimum free memory of gpus: {} MB'.format(min(gpu_resource)))

        SM = 500
        MD = 2500
        LG = 4500

        exceed_gpu_id = None
        # MB
        if model_size < 500 and min(gpu_resource) < SM:
            exceed_gpu_id = [i for i, e in enumerate(gpu_resource) if e < SM]
        elif 500 <= model_size < 2000 and min(gpu_resource) < MD:
            exceed_gpu_id = [i for i, e in enumerate(gpu_resource) if e < MD]
        elif 2000 <= model_size < 3000 and min(gpu_resource) < LG:
            exceed_gpu_id = [i for i, e in enumerate(gpu_resource) if e < LG]

        if exceed_gpu_id is not None:
            for idx in exceed_gpu_id:
                if not set_scale_0_least_recently_used(idx):
                    return {'message':'Fail to scale 0'}, 500
            return set_scale_model(model_name, scale, 1)

    if VERY_BUSY:
        return -1
    path = f'/models/{model_name}'
    params = {
        'min_worker': scale,
        'max_worker': scale,
        'synchronous': True
    }
    post_path = MANAGEMENT_URL + path
    VERY_BUSY = True
    res = requests.put(post_path, params=params)
    VERY_BUSY = False
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


def inference_model(model, data, path=None):
    path = (INFERENCE_URL if path is None else path) + f'/predictions/{model}'
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    res = requests.post(
         path, headers=headers, data=json.dumps(data)
    )

    if res.status_code != 200:
        return None

    return res.json()
