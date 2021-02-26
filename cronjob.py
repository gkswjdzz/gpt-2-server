from apscheduler.schedulers.background import BackgroundScheduler
from time import time
from torch_serve import set_scale_model


def write_info(model_name, infer_time):
    f = open('db.txt', 'a')
    f.write(f'{model_name},{infer_time}')
    f.close()


def runner():
    f = open('db.txt', 'r')
    lines = f.readlines()

    cur_time = int(time())

    needUpdate = False
    needUpdateModels = {}
    TIME = 50
    for line in lines:
        if line:
            model_name, latest_time = line.split(',')
            if cur_time - int(latest_time) > TIME:
                needUpdate = True
                set_scale_model(model_name, 0)
                print(f'stop {model_name}')
            else:
                if needUpdateModels[model_name] and needUpdateModels[model_name] < latest_time:
                    needUpdateModels[model_name] = latest_time
                    needUpdateModels = True
                else:
                    needUpdateModels[model_name] = latest_time 

    f.close()
    
    if needUpdate:
        f = open('db.txt', 'w')
        for key, value in needUpdateModels.items():
            f.write(f'{key},{value}')
        f.close()


def start_job():
    sched = BackgroundScheduler()
    sched.add_job(runner, 'interval', seconds=10)
    sched.start()
