import logging
import os
import re
import shutil
import sys

from tensorboardX import SummaryWriter

from utils.misc import time_str


def create_logger(logname, filename, level=logging.INFO, stream=True):
    l = logging.getLogger(logname)
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)4s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if stream:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        l.addHandler(sh)
    return l


class DistLogger(object):
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing

    def __del__(self):
        if self._lg is not None and hasattr(self._lg, 'close'):
            self._lg.close()


def create_loggers(job_cfg, dist):
    # create the exp dir
    if dist.is_master():
        os.makedirs(job_cfg.exp_root)
        
        # backup scripts
        back_dir = os.path.join(job_cfg.exp_root, 'back_up')
        shutil.copytree(
            src=job_cfg.prj_root, dst=back_dir,
            ignore=shutil.ignore_patterns('.*', '*ckpt*', '*exp*', '__pycache__'),
            ignore_dangling_symlinks=True
        )
        shutil.copytree(
            src=job_cfg.sh_root, dst=back_dir + job_cfg.sh_root.replace(job_cfg.prj_root, ''),
            ignore=lambda _, names: {n for n in names if not re.match(r'^(.*)\.(yaml|sh)$', n)},
            ignore_dangling_symlinks=True
        )
        print(f'{time_str()}[rk00] => All the scripts are backed up to \'{back_dir}\'.\n')
    dist.barrier()
    
    # create loggers
    logger = create_logger('G', os.path.join(job_cfg.exp_root, 'log.txt')) if dist.is_master() else None

    if dist.is_master():
        os.mkdir(os.path.join(job_cfg.exp_root, 'events'))
    dist.barrier()

    global_tensorboard_logger = SummaryWriter(os.path.join(job_cfg.exp_root, 'events', 'glb')) if dist.is_master() else None
    local_tensorboard_logger = SummaryWriter(os.path.join(job_cfg.exp_root, 'events', job_cfg.loc_desc))

    return (
        DistLogger(logger, verbose=dist.is_master()),
        DistLogger(global_tensorboard_logger, verbose=dist.is_master()),
        DistLogger(local_tensorboard_logger, verbose=True)
    )
