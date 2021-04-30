import argparse
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from logging import Logger
from pprint import pformat as pf
from typing import NamedTuple, Optional, List
import numpy as np

import colorama
import torch
import torch.nn.functional as F
from retrying import retry
from rsa.prime import is_prime
from tensorboardX import SummaryWriter
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

# from aug_op.rrc import CIFAR10PairTransform
from aug_op.aug import Augmenter
from aug_op.ops import GaussianBlur, RandSharpness
from meta import seatable_fname, run_shell_name
from model.moco import ModelMoCo
from utils.cfg import parse_cfg, Cfg, namedtuple_to_str
from utils.data import InputPairSet
from utils.dist import TorchDistManager
from utils.file import create_loggers
from utils.misc import time_str, filter_params, set_seed, AverageMeter, TopKHeap, adjust_learning_rate, accuracy, master_echo

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

# basic
parser.add_argument('--main_py_rel_path', type=str, required=True)
parser.add_argument('--exp_dirname', type=str, required=True)
parser.add_argument('--cfg', type=str, required=True)


def main():
    colorama.init(autoreset=True)
    args = parser.parse_args()
    
    sh_root = os.getcwd()
    job_name = os.path.split(sh_root)[-1]
    exp_root = os.path.join(sh_root, args.exp_dirname)
    os.chdir(args.main_py_rel_path)
    prj_root = os.getcwd()
    os.chdir(sh_root)
    
    dist = TorchDistManager(args.exp_dirname, 'auto', 'auto')

    descs = [f'rk{rk:02d}' for rk in range(dist.world_size)]
    local_desc = descs[dist.rank]
    job_kw = dict(
        sh_root=sh_root, job_name=job_name, exp_root=exp_root,
        prj_root=prj_root, descs=descs, local_desc=local_desc
    )
    cfg = parse_cfg(args.cfg, dist.rank, dist.world_size, job_kw)
    
    if dist.is_master():
        try:
            main_process(cfg, dist)
        except Exception as e:
            upd_seatable_file(exp_root, verbose=True, pr=-1., rem=0)
            raise e
    else:
        try:
            main_process(cfg, dist)
        except Exception:
            exit(-5)
    
    if isinstance(dist.WORLD_GROUP, int):
        dist.finalize()


seatable_kw = {}


@retry(stop_max_attempt_number=5, wait_fixed=500)
def upd_seatable_file(exp_root, verbose, **kw):
    if verbose:
        seatable_kw.update(kw)
        with open(os.path.join(exp_root, seatable_fname), 'w') as fp:
            json.dump([exp_root, seatable_kw], fp)
        

def main_process(cfg: Cfg, dist: TorchDistManager):
    # for i in range(dist.world_size):
    #     if i == dist.rank:
    #         print(f'[[[[ rk {dist.rank} ]]]]: dist.dev_idx={dist.dev_idx}, gpu_dev_idx={gpu_dev_idx}')
    #     dist.barrier()
    # assert dist.dev_idx == gpu_dev_idx
    
    on_imagenet = 'imagenet' in cfg.data.dataset
    sub_imagenet = on_imagenet and cfg.data.dataset != 'imagenet'

    lg, g_tb_lg, l_tb_lg = create_loggers(cfg.job, dist)
    lg: Logger = lg  # just for the code completion (actually is `DistLogger`)
    g_tb_lg: SummaryWriter = g_tb_lg  # just for the code completion (actually is `DistLogger`)
    l_tb_lg: SummaryWriter = l_tb_lg  # just for the code completion (actually is `DistLogger`)

    lg.info(f'=> [final cfg]:\n{namedtuple_to_str(cfg)}')
    
    if cfg.seed_base is None:
        lg.info(f'=> [main]: args.seed_base is None, no set_seed called')
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = False
    else:
        seeds = torch.zeros(dist.world_size).float()
        seeds[dist.rank] = cfg.seed
        dist.allreduce(seeds)
        dist.broadcast(seeds, 0)
        assert torch.allclose(seeds, torch.arange(cfg.seed_base, cfg.seed_base + dist.world_size).float())
        same_seed = cfg.torch_ddp
        set_seed(cfg.seed_base if same_seed else cfg.seed)
        lg.info(f'=> [main]: using {"the same seed" if same_seed else "diff seeds"}')
    
    upd_seatable_file(
        cfg.job.exp_root, dist.is_master(),
        gpu=dist.world_size if cfg.torch_ddp else 1,
        ds=cfg.data.dataset,
        # mom=args.moco_m,
        T=cfg.moco.moco_t,
        sbn=cfg.moco.sbn, mlp=cfg.moco.mlp, sym=cfg.moco.moco_symm, init=cfg.moco.init,
        ep=cfg.pretrain.epochs, bs=cfg.pretrain.batch_size, t_bs=cfg.pretrain.knn_ld_or_test_ld_batch_size, cos=cfg.pretrain.coslr, wp=cfg.pretrain.warmup, nowd=cfg.pretrain.nowd,
        v_ep=cfg.lnr_eval.eval_epochs, v_cos=cfg.lnr_eval.eval_coslr, v_wp=cfg.lnr_eval.eval_warmup, v_nowd=cfg.lnr_eval.eval_nowd,
        pr=0, rem=0, beg_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    auto_aug = None
    if on_imagenet:
        trans_ls = [
            transforms.RandomResizedCrop(224, scale=cfg.aug.rrc_range, ratio=cfg.aug.rrc_ratio),
            transforms.RandomApply([
                transforms.ColorJitter(*cfg.aug.cj_args)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(cfg.aug.blur_args) if cfg.aug.gblur else RandSharpness()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ]
        if cfg.aug.no_gray:
            trans_ls.pop(2)
        if cfg.aug.no_flip:
            trans_ls.pop(-3)
        
        if cfg.aug.auto_aug:
            trans_ls = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
            auto_aug = Augmenter(
                ch_means=cfg.data.meta.mean_std[0], ch_stds=cfg.data.meta.mean_std[1],
                adversarial=cfg.aug.adversarial,
                expansion=cfg.aug.expansion, act_name=cfg.aug.act_name,
                padding_mode=cfg.aug.padding_mode,
                target_norm=cfg.aug.target_norm, soft_target=cfg.aug.soft_target,
            ).cuda()
        
        pret_transform = transforms.Compose(trans_ls)
        lg.info(f'=> [pret transform] (auto_aug: {auto_aug is not None}): {pret_transform}')
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ])
        eval_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ])
    else:
        pret_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ])
        eval_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*cfg.data.meta.mean_std, inplace=True),
        ])
    
    # ds_choice = torch.randperm(8)[0]
    # dist.broadcast(ds_choice, 0)
    # ds_choice = ds_choice.item()
    # ds_root += f'_{ds_choice}'
    # master_echo(dist.is_master(), f'[dataset] choice={ds_choice}')
    
    set_kw = dict(root=cfg.data.ds_root, download=False)
    if sub_imagenet:
        set_kw['num_classes'] = cfg.data.meta.num_classes
    loader_kw = dict(num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_mem)
    dist_sp_kw = dict(num_replicas=dist.world_size, rank=dist.rank, shuffle=True)

    pret_iters = pret_ld = pret_sp = eval_iters = eval_ld = eval_sp = test_iters = test_ld = knn_iters = knn_ld = None
    for rk in range(dist.world_size):
        if rk == dist.rank:
            master_echo(True, f'{time_str()}[rk{dist.rank:2d}] construct dataloaders... ', tail='\\c')
            
            ds_clz = cfg.data.meta.clz
            pret_data = InputPairSet(ds_clz(train=True, transform=pret_transform, **set_kw))
            pret_sp = DistributedSampler(pret_data, **dist_sp_kw) if cfg.torch_ddp else None
            # todo: drop_last=True还会出现K不整除inp.shape[0]的情况吗？如果左下角的/mnt/lustre/tiankeyu/data_t1/moco_imn/exp/imn/200ep_cos_4gpu/实验没因为这个报error就说明没问题了
            pret_ld = DataLoader(pret_data, batch_size=cfg.pretrain.batch_size, sampler=pret_sp, shuffle=(pret_sp is None), drop_last=True, **loader_kw)
            pret_iters = len(pret_ld)
            lg.info(f'=> [main]: prepare pret_data (len={len(pret_data)}, bs={cfg.pretrain.batch_size}, iters={pret_iters}, ddp={cfg.torch_ddp}): @ {cfg.data.ds_root}')

            if not on_imagenet:
                knn_data = ds_clz(train=True, transform=test_transform, **set_kw)
                knn_ld = DataLoader(knn_data, batch_size=cfg.pretrain.knn_ld_or_test_ld_batch_size, shuffle=False, drop_last=False, **loader_kw)
                knn_iters = len(knn_ld)
                lg.info(f'=> [main]: prepare knn_data  (len={len(knn_data)}, bs={cfg.pretrain.knn_ld_or_test_ld_batch_size}, iters={knn_iters}, ddp=False for knn): @ {cfg.data.dataset}')
            
            test_data = ds_clz(train=False, transform=test_transform, **set_kw)
            test_ld = DataLoader(test_data, batch_size=cfg.pretrain.knn_ld_or_test_ld_batch_size, shuffle=False, drop_last=False, **loader_kw)
            test_iters = len(test_ld)
            lg.info(f'=> [main]: prepare test_data (len={len(test_data)}, bs={cfg.pretrain.knn_ld_or_test_ld_batch_size}, iters={test_iters}, ddp=False for test): @ {cfg.data.dataset}')
            
            eval_data = ds_clz(train=True, transform=eval_transform, **set_kw)
            eval_sp = DistributedSampler(eval_data, **dist_sp_kw) if cfg.torch_ddp else None
            eval_ld = DataLoader(eval_data, batch_size=cfg.lnr_eval.eval_batch_size, sampler=eval_sp, shuffle=(eval_sp is None), drop_last=False, **loader_kw)
            eval_iters = len(eval_ld)
            lg.info(f'=> [main]: prepare eval_data (len={len(eval_data)}, bs={cfg.lnr_eval.eval_batch_size}, iters={eval_iters}, ddp={cfg.torch_ddp}): @ {cfg.data.dataset}\n')
            
            master_echo(True, f'    finished!', '36', tail='')
        
        dist.barrier()
    
    lg.info(
        f'=> [main]: create the moco model: (ddp={cfg.torch_ddp})\n'
        f'     arch={cfg.moco.arch}, feature dim={cfg.moco.moco_dim}\n'
        f'     Q size={cfg.moco.moco_k}, ema mom={cfg.moco.moco_m}, moco T={cfg.moco.moco_t}\n'
        f'     sync bn={cfg.moco.sbn}, mlp={cfg.moco.mlp}, symmetric loss={cfg.moco.moco_symm}'
    )
    # create model
    model_kw = dict(
        lg=lg,
        on_imagenet=on_imagenet,
        torch_ddp=cfg.torch_ddp,
        arch=cfg.moco.arch,
        K=cfg.moco.moco_k,  # queue size
        m=cfg.moco.moco_m,  # ema momentum
        T=cfg.moco.moco_t,  # temperature
        sbn=cfg.moco.sbn,   # actually, SyncBatchNorm is not used in mocov2's official implementation
        mlp=cfg.moco.mlp,
        symmetric=cfg.moco.moco_symm,
        init=cfg.moco.init
    )
    pretrain_model = ModelMoCo(dim=cfg.moco.moco_dim, **model_kw)
    lnr_eval_model = ModelMoCo(dim=cfg.data.meta.num_classes, **model_kw)
    lnr_eval_encoder_q = lnr_eval_model.create_final_encoder_q()
    del lnr_eval_model
    
    if cfg.eval_resume_ckpt is None:
        l_tb_lg._verbose = dist.is_master() or (cfg.pret_verbose and not cfg.torch_ddp)
        if on_imagenet:
            pret_knn_args = None
        else:
            pret_knn_args = (knn_iters, knn_ld, cfg.moco.knn_k, cfg.moco.knn_t, knn_ld.dataset.targets)
        pret_res_str = pretrain(
            pret_knn_args, auto_aug, cfg.data.meta.num_classes, ExpMeta(
                cfg.torch_ddp, cfg.moco.arch, cfg.job.exp_root, os.path.split(cfg.job.exp_root)[-1], cfg.job.descs, cfg.log_freq, cfg.resume_ckpt,
                cfg.pretrain.epochs, float(cfg.pretrain.lr), float(cfg.pretrain.auglr), float(cfg.pretrain.wd), float(cfg.pretrain.augwd), cfg.pretrain.nowd, cfg.pretrain.coslr, cfg.pretrain.schedule, cfg.pretrain.warmup, cfg.pretrain.grad_clip, cfg.pretrain.aug_grad_clip
            ),
            lg, g_tb_lg, l_tb_lg, dist, pretrain_model, pret_iters, pret_ld, pret_sp, test_iters, test_ld
        )

        # broadcasting = ?
        # if broadcasting:
        #     src_rank = topk_accs.argmax().item()
        #     for _, param in pretrain_model.state_dict().items():
        #         dist.broadcast(param.data, src_rank)
        
        d = pretrain_model.encoder_q.state_dict()
        ks = deepcopy(list(d.keys()))
        for k in ks:
            if k.startswith('fc.'):
                del d[k]
        msg = lnr_eval_encoder_q.load_state_dict(d, strict=False)
        assert len(msg.unexpected_keys) == 0 and all(k.startswith('fc.') for k in msg.missing_keys)
    else:
        pret_res_str = '[resumed]'

    torch.cuda.empty_cache()
    l_tb_lg._verbose = dist.is_master() or not cfg.torch_ddp
    linear_eval(
        pret_res_str, cfg.data.meta.num_classes, ExpMeta(
            cfg.torch_ddp, cfg.moco.arch, cfg.job.exp_root, os.path.split(cfg.job.exp_root)[-1], cfg.job.descs, cfg.log_freq, cfg.eval_resume_ckpt,
            cfg.lnr_eval.eval_epochs, float(cfg.lnr_eval.eval_lr), 0., float(cfg.lnr_eval.eval_wd), 0., cfg.lnr_eval.eval_nowd, cfg.lnr_eval.eval_coslr, cfg.lnr_eval.eval_schedule, cfg.lnr_eval.eval_warmup, cfg.lnr_eval.eval_grad_clip, None
        ),
        lg, g_tb_lg, l_tb_lg, dist, lnr_eval_encoder_q, eval_iters, eval_ld, eval_sp, test_iters, test_ld
    )
    
    g_tb_lg.close(), l_tb_lg.close()


class ExpMeta(NamedTuple):
    # configs
    torch_ddp: bool
    arch: str
    exp_root: str
    exp_dirname: str
    descs: List[str]
    log_freq: int
    resume_ckpt: Optional[str]
    
    # hyperparameters
    epochs: int
    lr: float
    auglr: float
    wd: float
    augwd: float
    nowd: bool
    coslr: bool
    schedule: List[int]
    warmup: bool
    grad_clip: Optional[float]
    aug_grad_clip: Optional[float]


def pretrain(
        pretrain_knn_args,
        auto_aug: Optional[Augmenter],
        num_classes: int,
        meta: ExpMeta, lg: Logger, g_tb_lg: SummaryWriter, l_tb_lg: SummaryWriter,
        dist: TorchDistManager, pret_model: ModelMoCo,
        pret_iters: int, pret_ld: DataLoader, pret_dist_sp: DistributedSampler, te_iters: int, te_ld: DataLoader,
):
    if meta.torch_ddp:
        assert pretrain_knn_args is None
        pret_model: DistributedDataParallel = DistributedDataParallel(pret_model.cuda(), device_ids=[dist.dev_idx], output_device=dist.dev_idx)
    else:
        pret_model: ModelMoCo = pret_model.cuda()
    params = filter_params(pret_model) if meta.nowd else pret_model.parameters()
    params = list(filter(lambda p: p.requires_grad, params))
    optimizer = torch.optim.SGD(params, lr=meta.lr, weight_decay=meta.wd, momentum=0.9)
    aug_optimizer = torch.optim.AdamW(auto_aug.parameters(), lr=meta.auglr, weight_decay=meta.augwd)
    lg.info(f'=> [pretrain]: create op: model_cls={pret_model.__class__.__name__}, len(params)={len(params)}, max_lr={meta.lr}, max_alr={meta.auglr}, wd={meta.wd}, nowd={meta.nowd}, coslr={meta.coslr}, warm up={meta.warmup}')

    if not meta.coslr and meta.schedule is not None:
        assert len(meta.schedule) > 0
        sc = meta.schedule
        lg.info(f'=> [pretrain]: origin lr schedule={sc} (ep wise)')
        sc = sorted(sc)
        for i, milestone_epoch in enumerate(sc):
            sc[i] = milestone_epoch * pret_iters
        lg.info(f'=> [pretrain]: updated lr schedule={sc} (iter wise)')
        meta = meta._replace(schedule=sc)

    epoch_start = 0
    best_test_acc1 = -1e7
    tr_loss_mov_avg = 0
    topk_acc1s = TopKHeap(maxsize=max(1, round(meta.epochs * 0.1)))
    # todo: double-check ckpt loading and early returning
    if meta.resume_ckpt is not None:
        pret_resume = torch.load(meta.resume_ckpt, map_location='cpu')
        epoch_start = pret_resume['epoch'] + 1
        best_test_acc1 = pret_resume['best_test_acc1']
        tr_loss_mov_avg = pret_resume.get('tr_loss_mov_avg', -1)
        [topk_acc1s.push_q(x) for x in pret_resume['topk_acc1s']]
        
        lg.info(f'=> [pretrain]: ckpt loaded from {meta.resume_ckpt}, last_ep={epoch_start - 1}, ep_start={epoch_start}')
        pret_model.load_state_dict(pret_resume['pret_model'])
        
        lg.info(f'=> [pretrain]: load optimizer.state from {meta.resume_ckpt}')
        optimizer.load_state_dict(pret_resume['optimizer'])
        aug_optimizer.load_state_dict(pret_resume['aug_optimizer'])
    
    time.sleep(1 + 2 * dist.rank)
    epoch_speed = AverageMeter(3)
    tr_loss_avg = AverageMeter(pret_iters)
    tr_acc1_avg = AverageMeter(pret_iters)
    tr_acc5_avg = AverageMeter(pret_iters)
    avgs = (tr_loss_avg, tr_acc1_avg, tr_acc5_avg)
    loop_start_t = time.time()
    for epoch in range(epoch_start, meta.epochs):
        if meta.torch_ddp:
            pret_dist_sp.set_epoch(epoch)
        ep_str = f'%{len(str(meta.epochs))}d'
        ep_str %= epoch + 1
        ep_str = f'ep[{ep_str}/{meta.epochs}]'
        if epoch % 7 == 0 or epoch == meta.epochs - 1:
            em_t = time.time()
            torch.cuda.empty_cache()
            master_echo(dist.is_master(), f' @@@@@ {meta.exp_root} , ept_cc: {time.time() - em_t:.3f}s,      pre_be={best_test_acc1:5.2f}', '36')
        
        start_t = time.time()
        tr_loss: float = train_one_ep(True, 'pretrain', lg, g_tb_lg, l_tb_lg, dist, meta, epoch, ep_str, pret_iters, pret_ld, auto_aug, pret_model, params, optimizer, aug_optimizer, avgs)
        tr_loss_mov_avg = tr_loss if tr_loss_mov_avg == 0 else tr_loss_mov_avg * 0.99 + tr_loss * 0.01
        train_t = time.time()
        
        if pretrain_knn_args is not None:
            test_acc1 = knn_test(pretrain_knn_args, lg, l_tb_lg, dist, meta.log_freq, epoch, ep_str, te_iters, te_ld, pret_model.encoder_q, num_classes)
        else:
            test_acc1 = -tr_loss
        if test_acc1 > 0:
            l_tb_lg.add_scalar(f'pretrain/knn_acc1', test_acc1, epoch + 1)
        test_t = time.time()
        
        topk_acc1s.push_q(test_acc1)
        best_updated = test_acc1 > best_test_acc1
        state_dict = {
            'arch': meta.arch, 'epoch': epoch, 'pret_model': pret_model.state_dict(), 'optimizer': optimizer.state_dict(), 'aug_optimizer': aug_optimizer.state_dict(),
            'topk_acc1s': list(topk_acc1s), 'best_test_acc1': best_test_acc1, 'tr_loss_mov_avg': tr_loss_mov_avg,
        }
        if best_updated:
            best_test_acc1 = test_acc1
            torch.save(state_dict, os.path.join(meta.exp_root, f'pretrain_best.pth'))
        if epoch == meta.epochs - 1 and dist.is_master():
            last_ep_ckpt_name = f'pretrain_final_acc{test_acc1:5.2f}.pth' if test_acc1 > 0 else f'pretrain_final_loss{-test_acc1:.3f}.pth'
            torch.save(state_dict, os.path.join(meta.exp_root, last_ep_ckpt_name))
        
        remain_time, finish_time = epoch_speed.time_preds(meta.epochs - (epoch + 1))
        lg.info(
            f'=> {ep_str}: L={tr_loss:.4g}, te-acc={test_acc1:5.2f}, tr={train_t - start_t:.2f}s, te={test_t - train_t:.2f}s       best={best_test_acc1:5.2f}\n'
            f'    pretrain [{str(remain_time)}] ({finish_time})'
        )
        if dist.is_master():
            upd_seatable_file(
                meta.exp_root, dist.is_master(), pr=min((epoch + 1) / meta.epochs, 0.999), lr=f'{meta.lr:.1g}', alr=f'{meta.auglr:.1g}', knn_acc=-15750 if np.isnan(test_acc1) else test_acc1,
                rem=remain_time.seconds, end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds)),
            )
        
        epoch_speed.update(time.time() - start_t)
        if epoch == epoch_start:
            master_echo(True, f'[rk{dist.rank:2d}] barrier test')
            dist.barrier()
    
    topk_test_acc1 = sum(topk_acc1s) / len(topk_acc1s)
    if meta.torch_ddp:
        perform_dict_str = ''
        pret_res_str = (
            f' avg tr losses  {tr_loss_mov_avg:.3f}\n'
            f' mean-top acc1s @ {topk_test_acc1:.3f}\n'
            f' best     acc1s @ {best_test_acc1:.3f}'
        )
        seatable_acc = best_test_acc1
        seatable_mov_avg = tr_loss_mov_avg
    else:
        topk_accs = dist.dist_fmt_vals(topk_test_acc1, None)
        best_accs = dist.dist_fmt_vals(best_test_acc1, None)
        tr_loss_mov_avgs = dist.dist_fmt_vals(tr_loss_mov_avg, None)
        perform_dict_str = pf({
            des: f'topk={ta.item():.3f}, best={ba.item():.3f}'
            for des, ta, ba in zip(meta.descs, topk_accs, best_accs)
        })
        pret_res_str = (
            f' avg tr losses  {str(tr_loss_mov_avgs).replace(chr(10), " ")}\n'
            f' mean-top acc1s @ (max={topk_accs.max():.3f}, mean={topk_accs.mean():.3f}, std={topk_accs.std():.3f}) {str(topk_accs).replace(chr(10), " ")})\n'
            f' best     acc1s @ (max={best_accs.max():.3f}, mean={best_accs.mean():.3f}, std={best_accs.std():.3f}) {str(best_accs).replace(chr(10), " ")})'
        )
        seatable_acc = best_accs.mean().item()
        seatable_mov_avg = tr_loss_mov_avgs.mean().item()
    
    [g_tb_lg.add_scalar(f'pretrain/knn_best1', seatable_acc, e) for e in [epoch_start, meta.epochs]]
    dt = time.time() - loop_start_t
    lg.info(
        f'==> pretrain finished,'
        f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)'
        f' topk: {pf([round(x, 2) for x in topk_acc1s])}\n'
        f' performance: \n{perform_dict_str}\n{pret_res_str}'
    )
    
    if dist.is_master():
        upd_seatable_file(
            meta.exp_root, dist.is_master(), pr=0.999, rem=180,      # linear_eval
            knn_acc=seatable_acc, knn_L=seatable_mov_avg,
            end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        lines = pret_res_str.splitlines()
        strs = ''.join([f'# {l}\n' for l in lines])
        with open(run_shell_name, 'a') as fp:
            print(f'\n# pretrain {meta.exp_dirname}:\n{strs}', file=fp)
    return pret_res_str


def linear_eval(
        pret_res_str: str,
        num_classes: int,
        meta: ExpMeta, lg: Logger, g_tb_lg: SummaryWriter, l_tb_lg: SummaryWriter,
        dist: TorchDistManager, encoder_q: torch.nn.Module,
        eval_iters: int, eval_ld: DataLoader, eval_dist_sp: DistributedSampler, te_iters: int, te_ld: DataLoader,
):
    for name, p in encoder_q.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            p.requires_grad = False
    encoder_q.fc.weight.data.normal_(mean=0.0, std=0.01)
    encoder_q.fc.bias.data.zero_()
    
    if meta.torch_ddp:
        encoder_q: DistributedDataParallel = DistributedDataParallel(encoder_q.cuda(), device_ids=[dist.dev_idx], output_device=dist.dev_idx)
    else:
        encoder_q: torch.nn.Module = encoder_q.cuda()
    params = filter_params(encoder_q) if meta.nowd else encoder_q.parameters()
    params = list(filter(lambda p: p.requires_grad, params))
    optimizer = torch.optim.SGD(params, lr=meta.lr, weight_decay=meta.wd, momentum=0.9)
    lg.info(f'=> [lnr_eval]: create op: model_cls={encoder_q.__class__.__name__}, len(params)={len(params)}, max_lr={meta.lr}, wd={meta.wd}, nowd={meta.nowd}, coslr={meta.coslr}, warm up={meta.warmup}')

    if not meta.coslr and meta.schedule is not None:
        assert len(meta.schedule) > 0
        sc = meta.schedule
        lg.info(f'=> [lnr_eval]: origin lr schedule={sc} (ep wise)')
        sc = sorted(sc)
        for i, milestone_epoch in enumerate(sc):
            sc[i] = milestone_epoch * eval_iters
        lg.info(f'=> [lnr_eval]: updated lr schedule={sc} (iter wise)')
        meta = meta._replace(schedule=sc)

    epoch_start = 0
    best_test_acc1 = best_test_acc5 = -5
    tr_loss_mov_avg = 0
    topk_acc1s = TopKHeap(maxsize=max(1, round(meta.epochs * 0.1)))
    # todo: double-check ckpt loading and early returning
    if meta.resume_ckpt is not None:
        eval_resume = torch.load(meta.resume_ckpt, map_location='cpu')
        epoch_start = eval_resume['epoch'] + 1
        best_test_acc1 = eval_resume['best_test_acc1']
        best_test_acc5 = eval_resume['best_test_acc5']
        [topk_acc1s.push_q(x) for x in eval_resume['topk_acc1s']]
        
        lg.info(f'=> [lnr_eval]: ckpt loaded from {meta.resume_ckpt}, last_ep={epoch_start - 1}, ep_start={epoch_start}')
        encoder_q.load_state_dict(eval_resume['encoder_q'])
        
        lg.info(f'=> [lnr_eval]: load optimizer.state from {meta.resume_ckpt}')
        optimizer.load_state_dict(eval_resume['optimizer'])
    
    local_encoder_q = encoder_q.module if meta.torch_ddp else encoder_q
    initial_encoder_q_state = deepcopy(local_encoder_q.state_dict())
    for name, p in local_encoder_q.named_parameters():
        assert p.requires_grad == (name in ['fc.weight', 'fc.bias'])
    
    time.sleep(1 + 2 * dist.rank)
    epoch_speed = AverageMeter(3)
    tr_loss_avg = AverageMeter(eval_iters)
    tr_acc1_avg = AverageMeter(eval_iters)
    tr_acc5_avg = AverageMeter(eval_iters)
    avgs = (tr_loss_avg, tr_acc1_avg, tr_acc5_avg)
    loop_start_t = time.time()
    for epoch in range(epoch_start, meta.epochs):
        if meta.torch_ddp:
            eval_dist_sp.set_epoch(epoch)
        ep_str = f'%{len(str(meta.epochs))}d'
        ep_str %= epoch + 1
        ep_str = f'ep[{ep_str}/{meta.epochs}]'
        if epoch % 7 == 0 or epoch == meta.epochs - 1:
            em_t = time.time()
            torch.cuda.empty_cache()
            master_echo(dist.is_master(), f' @@@@@ {meta.exp_root} , ept_cc: {time.time() - em_t:.3f}s,      eva_be={best_test_acc1:5.2f}', '36')
        
        start_t = time.time()
        tr_loss: float = train_one_ep(False, 'lnr_eval', lg, g_tb_lg, l_tb_lg, dist, meta, epoch, ep_str, eval_iters, eval_ld, None, encoder_q, params, optimizer, None, avgs)
        tr_loss_mov_avg = tr_loss if tr_loss_mov_avg == 0 else tr_loss_mov_avg * 0.99 + tr_loss * 0.01
        train_t = time.time()
        
        test_acc1, test_acc5, test_loss = eval_test(lg, l_tb_lg, dist, meta.log_freq, epoch, ep_str, te_iters, te_ld, local_encoder_q, num_classes)
        l_tb_lg.add_scalar(f'lnr_eval/test_acc1', test_acc1, epoch + 1)
        l_tb_lg.add_scalar(f'lnr_eval/test_acc5', test_acc5, epoch + 1)
        l_tb_lg.add_scalar(f'lnr_eval/test_loss', test_loss, epoch + 1)
        test_t = time.time()
        
        topk_acc1s.push_q(test_acc1)
        best_updated = test_acc1 > best_test_acc1
        state_dict = {
            'arch': meta.arch, 'epoch': epoch, 'encoder_q': encoder_q.state_dict(), 'optimizer': optimizer.state_dict(),
            'topk_acc1s': list(topk_acc1s), 'best_test_acc1': best_test_acc1, 'best_test_acc5': best_test_acc5,
        }
        if best_updated:
            best_test_acc1 = test_acc1
            best_test_acc5 = test_acc5
            torch.save(state_dict, os.path.join(meta.exp_root, f'lnr_eval_best.pth'))
        
        remain_time, finish_time = epoch_speed.time_preds(meta.epochs - (epoch + 1))
        lg.info(
            f'=> {ep_str}: L={tr_loss:.4g}, te-acc={test_acc1:5.2f}, tr={train_t - start_t:.2f}s, te={test_t - train_t:.2f}s       best={best_test_acc1:5.2f}\n'
            f'    lnr_eval [{str(remain_time)}] ({finish_time})'
        )
        if dist.is_master():
            upd_seatable_file(
                meta.exp_root, dist.is_master(), pr=(epoch + 1) / meta.epochs, v_lr=f'{meta.lr:.1g}', test_acc=test_acc1,
                rem=remain_time.seconds, end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds)),
            )
        
        epoch_speed.update(time.time() - start_t)
        if epoch == epoch_start:
            master_echo(True, f'[rk{dist.rank:2d}] barrier test')
            dist.barrier()
            sanity_check(local_encoder_q.state_dict(), initial_encoder_q_state)
            for name, p in local_encoder_q.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    assert p.grad is None
            del initial_encoder_q_state
    
    topk_test_acc1 = sum(topk_acc1s) / len(topk_acc1s)
    if meta.torch_ddp:
        perform_dict_str = ''
        eval_str = (
            f' avg tr losses  {tr_loss_mov_avg:.3f}\n'
            f' best     acc5s @ {best_test_acc5:.3f}\n'
            f' mean-top acc1s @ {topk_test_acc1:.3f}\n'
            f' best     acc1s @ {best_test_acc1:.3f}'
        )
        seatable_acc = best_test_acc1
        seatable_mov_avg = tr_loss_mov_avg
    else:
        topk_accs = dist.dist_fmt_vals(topk_test_acc1, None)
        best_accs = dist.dist_fmt_vals(best_test_acc1, None)
        best_acc5s = dist.dist_fmt_vals(best_test_acc5, None)
        tr_loss_mov_avgs = dist.dist_fmt_vals(tr_loss_mov_avg, None)
        perform_dict_str = pf({
            des: f'topk={ta.item():.3f}, best={ba.item():.3f}'
            for des, ta, ba in zip(meta.descs, topk_accs, best_accs)
        })
        eval_str = (
            f' avg tr losses  {str(tr_loss_mov_avgs).replace(chr(10), " ")}\n'
            f' best     acc5s @ (max={best_acc5s.max():.3f}, mean={best_acc5s.mean():.3f}, std={best_acc5s.std():.3f}) {str(best_acc5s).replace(chr(10), " ")})\n'
            f' mean-top acc1s @ (max={topk_accs.max():.3f}, mean={topk_accs.mean():.3f}, std={topk_accs.std():.3f}) {str(topk_accs).replace(chr(10), " ")})\n'
            f' best     acc1s @ (max={best_accs.max():.3f}, mean={best_accs.mean():.3f}, std={best_accs.std():.3f}) {str(best_accs).replace(chr(10), " ")})'
        )
        seatable_acc = best_accs.mean().item()
        seatable_mov_avg = tr_loss_mov_avgs.mean().item()
    
    lg.info(f'==> pretrain.results: \n{pret_res_str}\n')
    [g_tb_lg.add_scalar(f'lnr_eval/test_best1', seatable_acc, e) for e in [epoch_start, meta.epochs]]
    dt = time.time() - loop_start_t
    lg.info(
        f'==> lnr_eval finished,'
        f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)'
        f' topk: {pf([round(x, 2) for x in topk_acc1s])}\n'
        f' performance: \n{perform_dict_str}\n{eval_str}'
    )
    
    if dist.is_master():
        upd_seatable_file(
            meta.exp_root, dist.is_master(), pr=1., rem=0,
            test_acc=seatable_acc, test_L=seatable_mov_avg,
            end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        lines = eval_str.splitlines()
        strs = ''.join([f'# {l}\n' for l in lines])
        with open(run_shell_name, 'a') as fp:
            print(f'\n# lnr_eval {meta.exp_dirname}:\n{strs}', file=fp)


# one epoch
def train_one_ep(is_pretrain, prefix, lg, g_tb_lg, l_tb_lg, dist, meta: ExpMeta, epoch, ep_str, tr_iters, tr_ld, auto_aug, dist_or_local_model, params, op, aug_op, avgs):
    using_auto_aug = auto_aug is not None
    if is_pretrain:
        dist_or_local_model.train()
        if using_auto_aug:
            auto_aug.train()
    else:
        dist_or_local_model.eval()    # todo: 全连接也弄成eval？确定吗？应该只是说提feature层应该不能更新吧，FC应该可以更新的吧！看下moco源代码！
    
    tr_loss_avg, tr_acc1_avg, tr_acc5_avg = avgs
    log_iters = tr_iters // meta.log_freq
    while log_iters > 1:
        if is_prime(log_iters):
            break
        log_iters -= 1
    
    tot_loss, tot_num = 0.0, 0
    last_t = time.time()
    
    for it, (data1, data2) in enumerate(tr_ld):
        cur_iter = it + epoch * tr_iters
        max_iter = meta.epochs * tr_iters
        
        data_t = time.time()
        bs = data1.shape[0]
        data1, data2 = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True)
        if using_auto_aug:
            (concated_aug_vec, normalized_vec1, normalized_vec2, aug_dim, norm_p), (data1, data2) = auto_aug(data1, normalizing=True)
        cuda_t = time.time()
        
        assert torch.isnan(data1).sum().item() == 0
        assert torch.isnan(data2).sum().item() == 0
        
        if is_pretrain:
            loss = dist_or_local_model(data1, data2)
        else:
            oup = dist_or_local_model(data1)
            loss = F.cross_entropy(oup, data2)
            acc1, acc5 = accuracy(oup, data2, topk=(1, 5))
            tr_acc1_avg.update(acc1, bs)
            tr_acc5_avg.update(acc5, bs)
        l = loss.item()
        tr_loss_avg.update(l, bs)
        tot_num += bs
        tot_loss += l * bs
        forw_t = time.time()
        
        op.zero_grad()
        if using_auto_aug:
            aug_op.zero_grad()
        loss.backward()
        back_t = time.time()
        
        if using_auto_aug:
            sche_aug_lr = adjust_learning_rate(aug_op, cur_iter, max_iter, meta.auglr, meta)
            if meta.aug_grad_clip is not None:
                orig_aug_norm = torch.nn.utils.clip_grad_norm_(auto_aug.parameters(), meta.aug_grad_clip)
                actual_aug_lr = sche_aug_lr * min(1, meta.aug_grad_clip / orig_aug_norm)
            else:
                orig_aug_norm = meta.aug_grad_clip
                actual_aug_lr = sche_aug_lr
        
        sche_lr = adjust_learning_rate(op, cur_iter, max_iter, meta.lr, meta)
        clipping = meta.grad_clip is not None and cur_iter < tr_iters * 10
        if clipping:
            orig_norm = torch.nn.utils.clip_grad_norm_(params, meta.grad_clip)
            actual_lr = sche_lr * min(1, meta.grad_clip / orig_norm)
        else:
            orig_norm = meta.grad_clip
            actual_lr = sche_lr
            
        clip_t = time.time()
        
        op.step()
        if using_auto_aug:
            auto_aug.inverse_grad()
            aug_op.step()
        step_t = time.time()

        if cur_iter < 10 or cur_iter % log_iters == 0 or (actual_lr < sche_lr - 1e-6 and random.randrange(8) == 0):
            g_tb_lg.add_scalars(f'{prefix}/lr', {'scheduled': sche_lr}, cur_iter)
            if clipping:
                g_tb_lg.add_scalar(f'{prefix}/orig_norm', orig_norm, cur_iter)
                g_tb_lg.add_scalars(f'{prefix}/lr', {'actual': actual_lr}, cur_iter)
        
        if tr_loss_avg.last > 12 or not is_pretrain and epoch < 2:
            l_tb_lg.add_scalar(f'{prefix}/train_loss', tr_loss_avg.last, cur_iter)
        
        if using_auto_aug:
            if cur_iter < 50 or cur_iter % log_iters == 0 or (actual_aug_lr < sche_aug_lr - 1e-6 and random.randrange(16) == 0):
                g_tb_lg.add_scalars(f'{prefix}/aug_lr', {'scheduled': sche_aug_lr}, cur_iter)
                g_tb_lg.add_scalar(f'{prefix}/orig_aug_grad_norm', orig_aug_norm, cur_iter)
                g_tb_lg.add_scalars(f'{prefix}/aug_lr', {'actual': actual_aug_lr}, cur_iter)
    
                k = max(round(concated_aug_vec.shape[0] * 0.3), 1)
                aug_vec1, aug_vec2 = concated_aug_vec.data[:, :aug_dim], concated_aug_vec.data[:, aug_dim:]
                aug_norm1, aug_norm2 = aug_vec1.norm(norm_p, dim=1).topk(k)[0].mean().item(), aug_vec2.norm(norm_p, dim=1).topk(k)[0].mean().item()
                aug_grad1, aug_grad2 = concated_aug_vec.grad[:, :aug_dim], concated_aug_vec.grad[:, aug_dim:]
                
                g_tb_lg.add_scalars(f'aug_vec_norm/before_normalize', {'o1': aug_norm1, 'o2': aug_norm2}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec_norm/after_normalize', {'o1': normalized_vec1.norm(norm_p, dim=1).topk(k)[0].mean().item(), 'o2': normalized_vec2.norm(norm_p, dim=1).topk(k)[0].mean().item()}, cur_iter)
                col_h1, col_s1, col_v1, blur1, tr_x1, tr_y1, area1, ratio1 = aug_vec1.unbind(1)
                col_h2, col_s2, col_v2, blur2, tr_x2, tr_y2, area2, ratio2 = aug_vec2.unbind(1)
                g_tb_lg.add_scalars(f'aug_vec/col_h', {'o1': col_h1.abs().topk(k)[0].mean().item(), 'o2': col_h2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/col_s', {'o1': col_s1.abs().topk(k)[0].mean().item(), 'o2': col_s2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/col_v', {'o1': col_v1.abs().topk(k)[0].mean().item(), 'o2': col_v2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/blur', {'o1': blur1.abs().topk(k)[0].mean().item(), 'o2': blur2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/tr_x', {'o1': tr_x1.abs().topk(k)[0].mean().item(), 'o2': tr_x2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/tr_y', {'o1': tr_y1.abs().topk(k)[0].mean().item(), 'o2': tr_y2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/area', {'o1': area1.abs().topk(k)[0].mean().item(), 'o2': area2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_vec/ratio', {'o1': ratio1.abs().topk(k)[0].mean().item(), 'o2': ratio2.abs().topk(k)[0].mean().item()}, cur_iter)
    
                col_h1, col_s1, col_v1, blur1, tr_x1, tr_y1, area1, ratio1 = aug_grad1.unbind(1)
                col_h2, col_s2, col_v2, blur2, tr_x2, tr_y2, area2, ratio2 = aug_grad2.unbind(1)
                g_tb_lg.add_scalars(f'aug_grad/grad_col_h', {'o1': col_h1.abs().topk(k)[0].mean().item(), 'o2': col_h2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_col_s', {'o1': col_s1.abs().topk(k)[0].mean().item(), 'o2': col_s2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_col_v', {'o1': col_v1.abs().topk(k)[0].mean().item(), 'o2': col_v2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_blur', {'o1': blur1.abs().topk(k)[0].mean().item(), 'o2': blur2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_tr_x', {'o1': tr_x1.abs().topk(k)[0].mean().item(), 'o2': tr_x2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_tr_y', {'o1': tr_y1.abs().topk(k)[0].mean().item(), 'o2': tr_y2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_area', {'o1': area1.abs().topk(k)[0].mean().item(), 'o2': area2.abs().topk(k)[0].mean().item()}, cur_iter)
                g_tb_lg.add_scalars(f'aug_grad/grad_ratio', {'o1': ratio1.abs().topk(k)[0].mean().item(), 'o2': ratio2.abs().topk(k)[0].mean().item()}, cur_iter)
        
        if cur_iter % log_iters == 0:
            # l_tb_lg.add_scalars(f'{prefix}/tr_loss', {'it': loss_avg.avg}, cur_iter)
            l_tb_lg.add_scalar(f'{prefix}/train_loss', tr_loss_avg.last, cur_iter)
            l_tb_lg.add_scalar(f'{prefix}/train_loss_ep', tr_loss_avg.avg, cur_iter)
            if is_pretrain:
                acc_str = ''
            else:
                acc_str = f'tr_a1={tr_acc1_avg.avg:5.2f}, tr_a5={tr_acc5_avg.avg:5.2f}'
                l_tb_lg.add_scalar(f'{prefix}/train_acc1', tr_acc1_avg.avg, cur_iter)
                l_tb_lg.add_scalar(f'{prefix}/train_acc5', tr_acc5_avg.avg, cur_iter)
            lg.info(
                f'  grad norm [{orig_norm:.2g}]\n'
                f'       {ep_str} it[{it + 1}/{tr_iters}]:'
                f' L={tr_loss_avg.avg:.4g} {acc_str}      '
                f' da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fo[{forw_t - cuda_t:.3f}], ba[{back_t - forw_t:.3f}],'
                f' cl[{clip_t - back_t:.3f}], op[{step_t - clip_t:.3f}]'
            )
        
        last_t = time.time()
    
    return tot_loss / tot_num


# test using a knn monitor
def knn_test(knn_args, lg, l_tb_lg, dist, log_freq, epoch, ep_str, te_iters, te_ld, local_encoder_q, num_classes):
    knn_iters, knn_ld, knn_k, knn_t, targets = knn_args
    # log_iters = te_iters // log_freq
    
    local_encoder_q.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for inp, _ in knn_ld:
            feature = local_encoder_q(inp.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        # last_t = time.time()
        for inp, tar in te_ld:
            # data_t = time.time()
            inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
            # cuda_t = time.time()
            feature = local_encoder_q(inp)
            feature = F.normalize(feature, dim=1)
            # fea_t = time.time()
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, knn_k, knn_t)
            # pred_t = time.time()
            
            total_num += inp.shape[0]
            total_top1 += (pred_labels[:, 0] == tar).float().sum().item()
            
            # if it % log_iters == 0:
            #     cur_te_acc1 = total_top1 / total_num * 100
            #     lg.info(
            #         f'     ep[{ep_str}] it[{it + 1}/{te_iters}]: *test acc={cur_te_acc1:5.2f}\n'
            #         f'       da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fe[{fea_t - cuda_t:.3f}], kn[{pred_t - fea_t:.3f}]'
            #     )
            
            # last_t = time.time()
    
    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()
    
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def eval_test(lg, l_tb_lg, dist, log_freq, epoch, ep_str, te_iters, te_ld, local_encoder_q, num_classes):
    local_encoder_q.eval()
    with torch.no_grad():
        tot_acc1, tot_acc5, tot_loss, tot_num = 0, 0, 0, 0
        for inp, tar in te_ld:
            bs = inp.shape[0]
            inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
            oup = local_encoder_q(inp)
            assert oup.shape[1] == num_classes
            loss = F.cross_entropy(oup, tar)
            acc1, acc5 = accuracy(oup, tar, topk=(1, 5))
            tot_acc1 += acc1 * bs
            tot_acc5 += acc5 * bs
            tot_loss += loss.item() * bs
            tot_num += bs
        
        return tot_acc1 / tot_num, tot_acc5 / tot_num, tot_loss / tot_num


def sanity_check(current_local_encoder_q_state, initial_local_encoder_q_state):
    ks1, ks2 = list(current_local_encoder_q_state.keys()), list(initial_local_encoder_q_state.keys())
    assert ks1 == ks2
    for key in filter(lambda k: not k.startswith('fc.'), ks1):
        t1, t2 = current_local_encoder_q_state[key], initial_local_encoder_q_state[key]
        t1: torch.Tensor
        t2: torch.Tensor
        assert (t1 == t2).all(), f'{key} changed in the linear evaluation'


if __name__ == '__main__':
    main()
