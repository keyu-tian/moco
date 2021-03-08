import os
import sys
import time

exp_dir_name = sys.argv[1]
exp_root = os.path.join(os.getcwd(), exp_dir_name)

while not os.path.exists(exp_root):
    print('[spying]')
    time.sleep(10)

