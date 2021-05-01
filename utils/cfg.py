import os
from collections import OrderedDict
from pprint import pformat
from typing import NamedTuple, Optional, List

import yaml
from easydict import EasyDict

from utils.data import DatasetMeta, dataset_metas


class JobCfg(NamedTuple):
    sh_root: str
    job_name: str
    exp_root: str
    prj_root: str
    descs: List[str]
    local_desc: str


class AugCfg(NamedTuple):
    no_gray: bool = False
    no_flip: bool = False
    rrc_range: list = [0.2, 1.]
    rrc_ratio: list = [3/4, 4/3]
    cj_args: list = [0.4, 0.4, 0.4, 0.1]
    blur_args: list = [.1, 2.]
    gblur: bool = True
    
    auto_aug: bool = False
    adversarial: bool = True
    expansion: int = 256
    act_name: str = 'tanh'    # 'tanh', 'swish'
    padding_mode: str = 'border'    # 'border', 'reflection' or 'zeros'
    target_norm: float = 1.1
    soft_target: float = 0.2
    

class DataCfg(NamedTuple):
    dataset: str = 'cifar10'
    meta: DatasetMeta = None
    ds_root: Optional[str] = None
    num_workers: int = 4
    pin_mem: bool = True


class MocoCfg(NamedTuple):
    arch: str = 'resnet18'
    init: bool = False
    moco_dim: int = 128  # cifar & imn: 128
    moco_k: float = 128  # cifar: 4096, imn: 65536
    moco_m: float = 0.99  # cifar: 0.99, imn: 0.999
    moco_t: float = 0.1  # cifar: 0.1, imn(mocov1): 0.07, v2: 0.2
    sbn: bool = False
    mlp: bool = True
    moco_symm: bool = True
    knn_k: int = 200  # only for cifar
    knn_t: float = 0.1  # only for cifar


class PretrainCfg(NamedTuple):
    epochs: int = 200
    batch_size: int = 512  # cifar: 512, imn: 256
    knn_ld_or_test_ld_batch_size: int = 512
    lr: float = 0.03  # cifar: 0.06 for b512, imn: 0.03 for b256
    augop: str = 'SGD'
    auglr: float = 0.001
    augmom: float = 0.3
    coslr: bool = True
    schedule: List[int] = [120, 160]  # step decay
    warmup: bool = True
    wd: float = 1e-4
    augwd: float = 1e-5
    nowd: bool = False
    grad_clip: Optional[float] = 5.
    aug_grad_clip: Optional[float] = 2.


class LinearEvalCfg(NamedTuple):
    eval_epochs: int = 200
    eval_batch_size: int = 512  # cifar: 512, imn: 256
    eval_lr: float = 30  # cifar & imn: 30 for b512 & b256
    eval_coslr: bool = True
    eval_schedule: List[int] = [60, 80]
    eval_warmup: bool = True
    eval_wd: float = 0.
    eval_nowd: bool = False
    eval_grad_clip: Optional[float] = 5.


class Cfg(NamedTuple):
    torch_ddp: bool = False
    resume_ckpt: Optional[str] = None
    eval_resume_ckpt: Optional[str] = None
    seed_base: Optional[int] = None
    seed: Optional[int] = None
    log_freq: int = 3
    pret_verbose: bool = False
    
    job: JobCfg = None
    aug: AugCfg = None
    data: DataCfg = None
    moco: MocoCfg = None
    pretrain: PretrainCfg = None
    lnr_eval: LinearEvalCfg = None


def parse_cfg(cfg_path, rank, world_size, job_kw) -> Cfg:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    
    # aug cfg
    if 'aug' in cfg:
        for key in ['rrc_range', 'rrc_ratio']:
            if key in cfg.aug:
                if isinstance(cfg.aug[key][0], str):
                    cfg.aug[key] = [eval(r) for r in cfg.aug[key]]
                assert len(cfg.aug[key]) == 2
                cfg.aug[key] = (min(cfg.aug[key]), max(cfg.aug[key]))
                if key == 'rrc_ratio':
                    assert abs(cfg.aug[key][0] * cfg.aug[key][1] - 1) < 1e-4
    
    # data cfg
    cfg.data.dataset = cfg.data.dataset.strip().lower()
    on_imagenet = 'imagenet' in cfg.data.dataset
    sub_imagenet = on_imagenet and cfg.data.dataset != 'imagenet'
    if sub_imagenet:
        num_classes = int(cfg.data.dataset.replace('imagenet', ''))
        dataset_meta = dataset_metas['subimagenet']
        dataset_meta = dataset_meta._replace(
            num_classes=num_classes,
            train_val_set_size=dataset_meta.train_val_set_size * num_classes,
            test_set_size=dataset_meta.test_set_size * num_classes,
        )
    else:
        dataset_meta = dataset_metas[cfg.data.dataset]
    if cfg.data.ds_root is None:
        cfg.data.ds_root = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', cfg.data.dataset))
    data_cfg = DataCfg(**cfg.data)
    data_cfg = data_cfg._replace(meta=dataset_meta)
    
    # moco
    cfg.moco.arch = cfg.moco.arch.strip().lower()
    
    # pretrain & linear eval
    if cfg.torch_ddp:
        assert cfg.pretrain.batch_size % world_size == 0 and cfg.lnr_eval.eval_batch_size % world_size == 0
        cfg.pretrain.batch_size //= world_size
        cfg.lnr_eval.eval_batch_size //= world_size
    if cfg.pretrain.schedule is not None and len(cfg.pretrain.schedule) > 0:
        cfg.pretrain.schedule = sorted(cfg.pretrain.schedule)
    if cfg.lnr_eval.eval_schedule is not None and len(cfg.lnr_eval.eval_schedule) > 0:
        cfg.lnr_eval.eval_schedule = sorted(cfg.lnr_eval.eval_schedule)
    
    # cfg
    if cfg.seed_base is not None:
        cfg.seed = cfg.seed_base + rank
    
    return Cfg(
        torch_ddp=cfg.torch_ddp,
        resume_ckpt=cfg.resume_ckpt,
        eval_resume_ckpt=cfg.eval_resume_ckpt,
        seed_base=cfg.seed_base,
        seed=cfg.seed,
        log_freq=cfg.log_freq,
        pret_verbose=cfg.pret_verbose,
        
        job=JobCfg(**job_kw),
        aug=AugCfg(**cfg.get('aug', {})),
        data=data_cfg,
        moco=MocoCfg(**cfg.moco),
        pretrain=PretrainCfg(**cfg.pretrain),
        lnr_eval=LinearEvalCfg(**cfg.lnr_eval),
    )


# https://stackoverflow.com/questions/16938456/serializing-a-nested-namedtuple-into-json-with-python-2-7
def _namedtuple_asdict(obj, ordered) -> dict:
    if hasattr(obj, '_asdict'):         # detect namedtuple
        d = OrderedDict(zip(obj._fields, (_namedtuple_asdict(item, ordered=ordered) for item in obj)))
        if not ordered:
            d = dict(d)
        return d
    else:
        return obj


def namedtuple_to_str(nt, ordered=True):
    clz = OrderedDict if ordered else dict
    return pformat(clz(_namedtuple_asdict(nt, ordered)))
