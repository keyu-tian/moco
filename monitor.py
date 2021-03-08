import datetime
import json
import os
import socket
import subprocess
import sys
import time

import colorama
from seatable_api import Base

from meta import seatable_fname

tag_choices = [
    'mlp', 'sbn', 'cos',
    'wp', 'nowd',
    'sym',
]


def get_ava_port():
    used_ports = os.popen("netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'").read()
    ava_ports = set(range(10000, 20000)) - set(map(int, used_ports.split()))
    return min(list(ava_ports), key=lambda x: -str(x).count('0'))


def record_dt(dd):
    dd['last_upd'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def create_or_upd_explore_table(base, abs_path=None, rid=None, **kwargs):
    t_name = 'explore'
    tags = []
    for k, v in kwargs.items():
        if k in tag_choices and v:
            tags.append(k)
            kwargs.pop(k)
    
    dd = dict(tags=tags, **kwargs)
    if abs_path is not None:
        dd['abs_path'] = abs_path
        dd['exp'] = '/'.join(abs_path.split('/')[-2:])
    record_dt(dd)
    
    if rid is None:
        return base.append_row(t_name, dd)['_id']
    else:
        base.update_row(t_name, rid, dd)
        return rid


def main():
    colorama.init(autoreset=True)
    ssl_aug_api_token = '669cdc2fc382cefb698d4c629dc2164e1f6772c5'
    server_url = 'https://cloud.seatable.cn'
    base = Base(ssl_aug_api_token, server_url)
    base.auth()
    
    exp_dir_name = sys.argv[1]
    exp_root = os.path.join(os.getcwd(), exp_dir_name)
    seatable_file = os.path.join(exp_root, seatable_fname)
    
    while not os.path.exists(seatable_file):
        print(colorama.Fore.GREEN + '[monitor] waiting for creating seatable temp file...')
        time.sleep(20)
    
    ava_port = get_ava_port()
    print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] found an ava port={ava_port}')
    cmd = f'tensorboard --logdir . --port {ava_port} --bind_all'
    sp = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, bufsize=-1)
    
    with open(seatable_file, 'r') as fp:
        last_dd = json.load(fp)
        last_dd['tb'] = f'{socket.gethostbyname(socket.gethostname())}:{ava_port}'
    
    with open(seatable_file, 'w') as fp:
        json.dump(last_dd, fp)
    
    try:
        rid = None
        while True:
            with open(seatable_file, 'r') as fp:
                dd = json.load(fp)
            if dd == last_dd:
                print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] same...')
                continue
            
            last_dd = dd
            abs_path, kwargs = dd
            print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] updating...')
            rid = create_or_upd_explore_table(base, abs_path, rid, **kwargs)
            print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] updated')
            time.sleep(5)
    
    except Exception as e:
        sp.kill()
        raise e
    
    sp.kill()


if __name__ == '__main__':
    main()
