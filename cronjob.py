from apscheduler.schedulers.background import BackgroundScheduler
from time import time
from torch_serve import set_scale_model

TIME = 3600


def write_info(model_name, infer_time):
    f = open('db.txt', 'a')
    f.write(f'{model_name},{infer_time}\n')
    f.close()


def runner():
    f = open('db.txt', 'r')
    lines = f.readlines()
    f.close()

    cur_time = int(time())

    need_update = False
    need_update_models = dict()

    for line in lines:
        if line:
            model_name, latest_time = line.split(',')
            if cur_time - int(latest_time) > TIME:
                need_update = True
                set_scale_model(model_name, 0)
                print(f'stop {model_name}')
            else:
                if model_name in need_update_models \
                        and need_update_models[model_name] < latest_time:
                    need_update_models[model_name] = latest_time
                    need_update = True
                else:
                    need_update_models[model_name] = latest_time


    if need_update:
        f = open('db.txt', 'w')
        need_update_models = sorted(need_update_models.items(), key=(lambda x: x[1]))
        for key, value in need_update_models:
            f.write(f'{key},{value}')
        f.close()


def start_job():
    sched = BackgroundScheduler()
    sched.add_job(runner, 'interval', seconds=10)
    sched.start()
