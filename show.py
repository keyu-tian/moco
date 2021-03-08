import json
import os
import socket
import subprocess
import time

import colorama

# from utils.seatable import fill_explore_table
from utils.seatable import fill_explore_table


def get_ava_port():
    used_ports = os.popen("netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'").read()
    ava_ports = set(range(10000, 20000)) - set(map(int, used_ports.split()))
    return min(list(ava_ports), key=lambda x: -str(x).count('0'))


def show():
    colorama.init(autoreset=True)
    ava_port = get_ava_port()
    
    print(colorama.Fore.LIGHTBLUE_EX + f'port={ava_port}')
    cmd = f'tensorboard --logdir . --port {ava_port} --bind_all'
    
    with open('.seatable_rid.json') as fp:
        rid = json.load(fp)
    fill_explore_table(abs_path=None, rid=rid, tb=f'{socket.gethostbyname(socket.gethostname())}:{ava_port}')
    
    sp = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, bufsize=-1)
    cnt = 0
    while True:
        cnt += 1
        time.sleep(60)
        if sp.poll() == 0:
            break


if __name__ == '__main__':
    show()
