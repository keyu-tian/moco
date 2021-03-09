import argparse
import json
import math
import os
import time
from datetime import datetime
from functools import partial
from logging import Logger
from pprint import pformat as pf

import colorama
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from meta import seatable_fname, run_shell_name
from model import model_entry
from model.bn import SplitBatchNorm
from utils.dist import TorchDistManager
from utils.file import create_files
from utils.misc import time_str, filter_params, set_seed, init_params, AverageMeter, MaxHeap

parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

# basic
parser.add_argument('--torch_ddp', action='store_true', help='using DistributedDataParallel')
parser.add_argument('--main_py_rel_path', type=str, required=True)
parser.add_argument('--exp_dirname', type=str, required=True)
parser.add_argument('--resume_ckpt', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--seed_base', default=0, type=int)
parser.add_argument('--log_freq', default=2, type=int)

# moco
parser.add_argument('--arch', default='resnet18')
parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco_k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature')
# parser.add_argument('--bn_splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
parser.add_argument('--sbn', action='store_true', help='use synchronized batchnorm')
parser.add_argument('--mlp', action='store_true', help='use mlp')
parser.add_argument('--moco_symm', action='store_true', help='use a symmetric loss function that backprops to both crops')

# training
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
# lr: 0.06 for batch 512 (or 0.03 for batch 256)
parser.add_argument('--lr', '--learning_rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --coslr is on')
parser.add_argument('--coslr', action='store_true', help='use cosine lr schedule')
parser.add_argument('--warmup', action='store_true', help='use warming up')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--nowd', action='store_true', help='no wd for params of bn and bias')

# data
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
parser.add_argument('--ds_root', default='', help='dataset root')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--pin_mem', action='store_true')

# knn monitor
parser.add_argument('--knn_k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn_t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')


# # set command line arguments here when running in ipynb
# args.epochs = 200
# args.coslr = True
# args.schedule = []  # coslr in use
# args.symmetric = True
# if args.results_dir == '':
#     args.results_dir = f'/content/drive/MyDrive/moco/moco_on_cifar10/raw_exp-{cur_dt_str()}'
#
# pp(vars(args))


class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        im_1 = self.transform(img)
        im_2 = self.transform(img)
        
        return im_1, im_2


class ModelMoCo(nn.Module):
    def __init__(self, lg, torch_ddp=False, arch='resnet18', dim=128, K=4096, m=0.99, T=0.1, sbn=False, mlp=False, symmetric=True):
        super(ModelMoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        
        # create the encoders
        # todo: torch_ddp
        assert not torch_ddp
        bn_splits = 1 if sbn else 8
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        self.encoder_q = model_entry(model_name=arch, num_classes=dim, norm_layer=norm_layer)
        self.encoder_k = model_entry(model_name=arch, num_classes=dim, norm_layer=norm_layer)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        
        init_params(self.encoder_q, output=lg.info)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer
        
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()
        
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        return x[idx_shuffle], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]
    
    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        
        return loss, q, k
    
    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        
        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)
        
        self._dequeue_and_enqueue(k)
        
        return loss


def adjust_learning_rate(optimizer, cur_iter, max_iter, max_lr, args):
    """Decay the learning rate based on schedule"""
    warmup_iters = max_iter // 100
    if args.warmup and cur_iter <= warmup_iters:
        ratio = cur_iter / warmup_iters
        base_lr = max_lr / 5
        lr = base_lr + ratio * (max_lr - base_lr)
    
    elif args.coslr:  # cosine lr schedule
        if args.warmup:
            ratio = (cur_iter - warmup_iters) / (max_iter - 1 - warmup_iters)
        else:
            ratio = cur_iter / (max_iter - 1)
        lr = max_lr * 0.5 * (1. + math.cos(math.pi * ratio))
    else:  # stepwise lr schedule
        lr = max_lr
        for milestone in args.schedule:
            lr *= 0.1 if cur_iter / max_iter >= milestone else 1.
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


# train for one epoch
def train(lg, l_tb_lg, dist, args, epoch, ep_str, iters_per_ep, moco_model, train_ld, train_op, tr_loss_avg):
    moco_model.train()
    log_iters = iters_per_ep // args.log_freq
    
    total_loss, total_num = 0.0, 0
    last_t = time.time()
    for it, (im_1, im_2) in enumerate(train_ld):
        data_t = time.time()
        cur_iter = it + epoch * iters_per_ep
        max_iter = args.epochs * iters_per_ep
        adjust_learning_rate(train_op, cur_iter, max_iter, args.lr, args)
        
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        cuda_t = time.time()
        
        loss = moco_model(im_1, im_2)
        forw_t = time.time()
        
        train_op.zero_grad()
        loss.backward()
        train_op.step()
        back_t = time.time()
        
        total_num += train_ld.batch_size
        total_loss += loss.item() * train_ld.batch_size
        
        cur_avg_loss = total_loss / total_num
        tr_loss_avg.update(cur_avg_loss)
        if cur_iter % log_iters == 0:
            l_tb_lg.add_scalars('pretrain/tr_loss', {'it': tr_loss_avg.avg}, cur_iter)
            lg.info(
                f'     ep[{ep_str}] it[{it + 1}/{iters_per_ep}]: L={cur_avg_loss:.2f} ({tr_loss_avg.avg:.2f})\n'
                f'       da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fo[{forw_t - cuda_t:.3f}], ba[{back_t - forw_t:.3f}]'
            )
        
        last_t = time.time()
    
    return total_loss / total_num


# test using a knn monitor
def test(lg, l_tb_lg, dist, args, epoch, ep_str, iters_per_ep, moco_encoder_q, knn_ld, test_ld):
    log_iters = iters_per_ep // args.log_freq
    
    moco_encoder_q.eval()
    num_classes = len(knn_ld.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for it, (data, target) in enumerate(knn_ld):
            feature = moco_encoder_q(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(knn_ld.dataset.targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        last_t = time.time()
        for it, (data, target) in enumerate(test_ld):
            data_t = time.time()
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            cuda_t = time.time()
            feature = moco_encoder_q(data)
            feature = F.normalize(feature, dim=1)
            fea_t = time.time()
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, args.knn_k, args.knn_t)
            knn_t = time.time()
            
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            
            if it % log_iters == 0:
                cur_te_acc1 = total_top1 / total_num * 100
                lg.info(
                    f'     ep[{ep_str}] it[{it + 1}/{iters_per_ep}]: *test acc={cur_te_acc1:5.3f}\n'
                    f'       da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fe[{fea_t - cuda_t:.3f}], kn[{knn_t - fea_t:.3f}]'
                )
            
            last_t = time.time()
    
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


def main():
    colorama.init(autoreset=True)
    args = parser.parse_args()
    args.dataset = args.dataset.strip().lower()
    
    args.sh_root = os.getcwd()
    args.job_name = os.path.split(args.sh_root)[-1]
    args.exp_root = os.path.join(args.sh_root, args.exp_dirname)
    os.chdir(args.main_py_rel_path)
    args.prj_root = os.getcwd()
    os.chdir(args.sh_root)
    
    dist = TorchDistManager('auto', 'auto')
    
    main_worker(args, dist)


def save_seatable_file(exp_root, kwargs):
    with open(os.path.join(exp_root, seatable_fname), 'w') as fp:
        json.dump([exp_root, kwargs], fp)


def main_worker(args, dist: TorchDistManager):
    # for i in range(dist.world_size):
    #     if i == dist.rank:
    #         print(f'[[[[ rk {dist.rank} ]]]]: dist.dev_idx={dist.dev_idx}, gpu_dev_idx={gpu_dev_idx}')
    #     dist.barrier()
    # assert dist.dev_idx == gpu_dev_idx
    
    descriptions = [f'rk{rk:2d}' for rk in range(dist.world_size)]
    # todo: change desc when doing a grid search
    
    args.loc_desc = descriptions[dist.rank]
    lg, g_tb_lg, l_tb_lg = create_files(args, dist)
    lg: Logger = lg  # just for the code completion (actually is `DistLogger`)
    g_tb_lg: SummaryWriter = g_tb_lg  # just for the code completion (actually is `DistLogger`)
    l_tb_lg: SummaryWriter = l_tb_lg  # just for the code completion (actually is `DistLogger`)
    lg.info(f'{time_str()} => [args]:\n{pf(vars(args))}\n')
    
    seeds = torch.zeros(dist.world_size).float()
    seeds[dist.rank] = args.seed = args.seed_base + dist.rank
    dist.allreduce(seeds)
    dist.broadcast(seeds, 0)
    assert torch.allclose(seeds, torch.arange(args.seed_base, args.seed_base + dist.world_size).float())
    same_seed = args.torch_ddp
    set_seed(args.seed_base if same_seed else args.seed)
    lg.info(f'=> [seed]: using {"the same seed" if same_seed else "diff seeds"}')

    if dist.is_master():
        seatable_kwargs = dict(
            ds=args.dataset, ep=args.epochs, bs=args.batch_size,
            mom=args.moco_m, T=args.moco_t,
            sbn=args.sbn, mlp=args.mlp, sym=args.moco_symm,
            cos=args.coslr, wp=args.warmup, nowd=args.nowd,
            pr=0, rem=0, beg_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )
        save_seatable_file(args.exp_root, seatable_kwargs)
    else:
        seatable_kwargs = None
    
    assert args.dataset == 'cifar10'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    ds_root = args.ds_root or os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', args.dataset))
    
    assert not args.torch_ddp
    lg.info(f'=> [create]: create train_ds: {args.dataset} (ddp={args.torch_ddp})')
    train_data = CIFAR10Pair(root=ds_root, train=True, transform=train_transform, download=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    lg.info(f'=> [create]: create knn_ds: {args.dataset} (ddp={args.torch_ddp})')
    knn_data = CIFAR10(root=ds_root, train=True, transform=test_transform, download=False)
    knn_loader = DataLoader(knn_data, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
    lg.info(f'=> [create]: create test_ds: {args.dataset} (ddp={args.torch_ddp})')
    test_data = CIFAR10(root=ds_root, train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    
    lg.info(f'=> [create]: create eval_ds: {args.dataset} (ddp={args.torch_ddp})')
    # eval_data = CIFAR10(root=ds_root, train=True, transform={['todo']}, download=False)
    # eval_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    
    lg.info(
        f'=> [create]: create moco model: (ddp={args.torch_ddp})\n'
        f'     arch={args.arch}, feature dim={args.moco_dim}\n'
        f'     q size={args.moco_k}, ema mom={args.moco_m}, T={args.moco_t}\n'
        f'     sync bn={args.sbn}, mlp={args.mlp}, moco_symm={args.moco_symm}'
    )
    # create model
    model = ModelMoCo(
        lg=lg,
        torch_ddp=args.torch_ddp,
        arch=args.arch,
        dim=args.moco_dim,
        K=args.moco_k,  # queue size
        m=args.moco_m,  # ema momentum
        T=args.moco_t,  # temperature
        sbn=args.sbn,
        mlp=args.mlp,
        symmetric=args.moco_symm,
    ).cuda()
    # print(model.encoder_q)
    
    # define optimizer
    lg.info(f'\n=> [create]: create op: max_lr={args.lr}, wd={args.wd}, nowd={args.nowd}, coslr={args.coslr}, warm up={args.warmup}')
    optimizer = torch.optim.SGD(filter_params(model) if args.nowd else model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    
    # load model if resume
    epoch_start = 0
    if args.resume_ckpt is not None:
        checkpoint = torch.load(args.resume_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume_ckpt))
    
    # pretraining
    start_pretrain_t = time.time()
    best_knn_acc1 = 0
    topk_acc1s = MaxHeap(maxsize=max(1, round(args.epochs * 0.05)))
    tr_iters_per_ep, te_iters_per_ep = len(train_loader), len(test_loader)
    epoch_speed = AverageMeter(3)
    tr_loss_avg = AverageMeter(tr_iters_per_ep)
    for epoch in range(epoch_start, args.epochs):
        ep_str = f'%{len(str(args.epochs))}d' % (epoch + 1)
        if epoch % 5 == 0 and dist.is_master():
            print(colorama.Fore.CYAN + f'@@@@@ {args.exp_root}')
            torch.cuda.empty_cache()
        
        start_t = time.time()
        tr_loss = train(lg, l_tb_lg, dist, args, epoch, ep_str, tr_iters_per_ep, model, train_loader, optimizer, tr_loss_avg)
        train_t = time.time()
        l_tb_lg.add_scalars('pretrain/tr_loss', {'ep': tr_loss}, (epoch + 1) * tr_iters_per_ep)
        
        knn_acc1 = test(lg, l_tb_lg, dist, args, epoch, ep_str, te_iters_per_ep, model.encoder_q, knn_loader, test_loader)
        topk_acc1s.push_q(knn_acc1)
        best_knn_acc1 = max(best_knn_acc1, knn_acc1)
        test_t = time.time()
        l_tb_lg.add_scalar('pretrain/knn_acc1', knn_acc1, epoch)
        
        # torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), }, args.results_dir + '/model_last.pth')
        
        remain_time, finish_time = epoch_speed.time_preds(args.epochs - (epoch + 1))
        lg.info(
            f'=> [ep {ep_str}/{args.epochs}]: L={tr_loss:.2f}, acc={knn_acc1:5.2f}, tr={train_t - start_t:.2f}s, te={test_t - train_t:.2f}s       best={best_knn_acc1:5.2f}\n'
            f'   [{str(remain_time)}] ({finish_time})'
        )
        if dist.is_master():
            seatable_kwargs.update(dict(
                knn_acc=knn_acc1, pr=(epoch + 1) / args.epochs, rem=remain_time.seconds,
                end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds))
            ))
            save_seatable_file(args.exp_root, seatable_kwargs)
        
        epoch_speed.update(time.time() - start_t)
        if epoch == epoch_start:
            print(f'[rk{dist.rank:2d}] barrier test')
            dist.barrier()
    
    topk_knn_acc1 = sum(topk_acc1s) / len(topk_acc1s)
    dt = time.time() - start_pretrain_t
    if not args.torch_ddp:
        topk_accs = dist.dist_fmt_vals(topk_knn_acc1, None)
        best_accs = dist.dist_fmt_vals(best_knn_acc1, None)
        perform_dict = pf({
            des: f'topk={ta.item():.3f}, best={ba.item():.3f}'
            for des, ta, ba in zip(descriptions, topk_accs, best_accs)
        })
        res_str = (
            f' mean-top accs @ (min={topk_accs.min():.3f}, mean={topk_accs.mean():.3f}, std={topk_accs.std():.3f}) {str(topk_accs).replace(chr(10), " ")})\n'
            f' best     accs @ (min={best_accs.min():.3f}, mean={best_accs.mean():.3f}, std={best_accs.std():.3f}) {str(best_accs).replace(chr(10), " ")})'
        )
        lg.info(
            f'==> pre-training finished,'
            f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)'
            f' topk: {pf([round(x, 2) for x in topk_acc1s])}\n'
            f' performance: \n{perform_dict}\n{res_str}'
        )
        
        if dist.is_master():
            seatable_kwargs.update(dict(
                knn_acc=best_accs.mean().item(), pr=1., rem=0,
                end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            save_seatable_file(args.exp_root, seatable_kwargs)
            ra, rb = res_str.splitlines()
            with open(run_shell_name, 'a') as fp:
                print(
                    f'# pretrain {args.exp_dirname}:\n'
                    f'# {ra}\n'
                    f'# {rb}\n'
                    , file=fp
                )
    
    else:
        assert False
    
    # linear evaluation
    lg.info('\n\n\n')
    epoch_speed = AverageMeter(3)
    
    g_tb_lg.close()
    l_tb_lg.close()
    # dist.finalize()


if __name__ == '__main__':
    main()
