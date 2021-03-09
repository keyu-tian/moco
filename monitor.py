import datetime
import json
import os
import random
import socket
import subprocess
import sys
import time
from copy import deepcopy

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


def create_or_upd_explore_table(base, abs_path, rid=None, **kwargs):
    assert abs_path is not None
    
    t_name = 'explore'
    tags = []
    new_kw = deepcopy(kwargs)
    for k, v in kwargs.items():
        if k in tag_choices and v:
            tags.append(k)
            new_kw.pop(k)
    
    kwargs = new_kw
    dd = dict(tags=tags, **kwargs)
    record_dt(dd)
    dd['abs_path'] = abs_path
    dd['exp'] = '/'.join(abs_path.split('/')[-2:])

    if rid is None:
        q = base.filter(t_name, f"exp='{dd['exp']}'")
        if q.exists():
            q.update(dd)
            return q.get()['_id'], False
        else:
            return base.append_row(t_name, dd)['_id'], True
    else:
        try:
            base.update_row(t_name, rid, dd)
            ret = rid, False
        except ConnectionError:
            ret = base.append_row(t_name, dd)['_id'], True
        return ret


def main():
    colorama.init(autoreset=True)
    ssl_aug_api_token = '669cdc2fc382cefb698d4c629dc2164e1f6772c5'
    server_url = 'https://cloud.seatable.cn'
    base = Base(ssl_aug_api_token, server_url)
    base.auth()
    
    exp_dir_name = sys.argv[1]
    exp_root = os.path.join(os.getcwd(), exp_dir_name)
    seatable_file = os.path.join(exp_root, seatable_fname)
    terminate_file = f'{exp_root}.terminate'
    
    while not os.path.exists(seatable_file):
        time.sleep(120)
        print(colorama.Fore.GREEN + f'[monitor] waiting for the seatable file at {seatable_file} ...')
        if os.path.exists(terminate_file):
            os.remove(terminate_file)
            print(colorama.Fore.CYAN + '[monitor] terminated.')
            exit(-1)
    
    ava_port = get_ava_port()
    print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] found an ava port={ava_port}')
    cmd = f'tensorboard --logdir . --port {ava_port} --bind_all'
    sp = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, bufsize=-1)
    tb_ip_port = f'{socket.gethostbyname(socket.gethostname())}:{ava_port}'
    
    with open(seatable_file, 'r') as fp:
        last_dd = json.load(fp)
    
    try:
        rid = None
        while True:
            if os.path.exists(terminate_file):
                os.remove(terminate_file)
                print(colorama.Fore.CYAN + '[monitor] terminated; use `sh ./kill.sh` to kill the tensorboard proc')
                exit(-1)    # sp will become an orphan process; use `sh ./kill.sh` to kill it
                
            time.sleep(15)
            attempts, max_att = 0, 5
            while attempts < max_att:
                try:
                    with open(seatable_file, 'r') as fp:
                        dd = json.load(fp)
                except json.decoder.JSONDecodeError:
                    attempts += 1
                else:
                    break
            
            if attempts == max_att:
                raise json.decoder.JSONDecodeError
            
            if dd == last_dd:
                # print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] same...')
                continue
            
            last_dd = dd
            abs_path, kwargs = dd
            rid, created = create_or_upd_explore_table(base, abs_path, rid, tb=tb_ip_port, **kwargs)
            if random.randrange(8) == 0 or created:
                print(colorama.Fore.LIGHTBLUE_EX + f'[monitor] {"created" if created else "updated"}')
    
    except Exception as e:
        sp.kill()
        raise e
    
    sp.kill()


if __name__ == '__main__':
    main()
