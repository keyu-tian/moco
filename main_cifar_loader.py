# import argparse
# import json
# import os
# import time
# from copy import deepcopy
# from datetime import datetime
# import random
# from logging import Logger
# from pprint import pformat as pf
# from typing import NamedTuple, Optional, List, Union, Iterator
#
# import colorama
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from tensorboardX import SummaryWriter
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import CIFAR10
#
# from aug_op.ops import *
# from meta import seatable_fname, run_shell_name
# from model.moco import ModelMoCo
# from utils.data import InfiniteBatchSampler
# from utils.dist import TorchDistManager
# from utils.file import create_files
# from utils.misc import time_str, filter_params, set_seed, AverageMeter, TopKHeap, adjust_learning_rate, accuracy
#
# parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
#
# # basic
# parser.add_argument('--torch_ddp', action='store_true', help='using DistributedDataParallel')
# parser.add_argument('--main_py_rel_path', type=str, required=True)
# parser.add_argument('--exp_dirname', type=str, required=True)
# parser.add_argument('--resume_ckpt', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--eval_resume_ckpt', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--seed_base', default=None, type=int)
# parser.add_argument('--log_freq', default=2, type=int)
#
# # moco
# parser.add_argument('--arch', default='resnet18')
# parser.add_argument('--init', action='store_true')
# parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
# parser.add_argument('--moco_k', default=4096, type=int, help='queue size; number of negative keys')
# parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
# parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature')
# # parser.add_argument('--bn_splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
# parser.add_argument('--sbn', action='store_true', help='use synchronized batchnorm')
# parser.add_argument('--mlp', action='store_true', help='use mlp')
# parser.add_argument('--moco_symm', action='store_true', help='use a symmetric loss function that backprops to both crops')
#
# # pretraining
# parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
# # lr: 0.06 for batch 512 (or 0.03 for batch 256)
# parser.add_argument('--lr', default=0.06, type=float, metavar='LR', help='initial learning rate')
# parser.add_argument('--coslr', action='store_true', help='use cosine lr schedule')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --coslr is on')
# parser.add_argument('--warmup', action='store_true', help='use warming up')
# parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
# parser.add_argument('--nowd', action='store_true', help='no wd for params of bn and bias')
# parser.add_argument('--grad_clip', default=5, type=float, help='max grad norm')
#
# # linear evaluation
# parser.add_argument('--eval_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--eval_batch_size', default=512, type=int, metavar='N', help='mini-batch size')
# # lr: 0.06 for batch 512 (or 0.03 for batch 256)
# parser.add_argument('--eval_lr', default=30, type=float, metavar='LR', help='initial learning rate')
# parser.add_argument('--eval_coslr', action='store_true', help='use cosine lr schedule')
# parser.add_argument('--eval_schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --coslr is on')
# parser.add_argument('--eval_warmup', action='store_true', help='use warming up')
# parser.add_argument('--eval_wd', default=0., type=float, metavar='W', help='weight decay')
# parser.add_argument('--eval_nowd', action='store_true', help='no wd for params of bn and bias')
# parser.add_argument('--eval_grad_clip', default=5, type=float, help='max grad norm')
#
# # data
# parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
# parser.add_argument('--ds_root', default='', help='dataset root')
# parser.add_argument('--num_workers', default=4, type=int)
# parser.add_argument('--pin_mem', action='store_true')
#
# # knn monitor
# parser.add_argument('--knn_k', default=200, type=int, help='k in kNN monitor')
# parser.add_argument('--knn_t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
#
# # explore
# parser.add_argument('--swap_iters', default=None, type=int)
# parser.add_argument('--adversarial', action='store_true')
# # todo: adversarial
#
#
# class CIFAR10Pair(CIFAR10):
#     def __getitem__(self, index):
#         img = self.data[index]
#         img = Image.fromarray(img)
#         im_1 = self.transform(img)
#         im_2 = self.transform(img)
#
#         return im_1, im_2
#
#
# def main():
#     colorama.init(autoreset=True)
#     args = parser.parse_args()
#     args.dataset = args.dataset.strip().lower()
#
#     args.sh_root = os.getcwd()
#     args.job_name = os.path.split(args.sh_root)[-1]
#     args.exp_root = os.path.join(args.sh_root, args.exp_dirname)
#     os.chdir(args.main_py_rel_path)
#     args.prj_root = os.getcwd()
#     os.chdir(args.sh_root)
#
#     dist = TorchDistManager('auto', 'auto')
#
#     main_process(args, dist)
#
#
# seatable_kw = {}
#
#
# def upd_seatable_file(exp_root, dist, **kw):
#     seatable_kw.update(kw)
#     if dist.is_master():
#         with open(os.path.join(exp_root, seatable_fname), 'w') as fp:
#             json.dump([exp_root, seatable_kw], fp)
#
#
# def main_process(args, dist: TorchDistManager):
#     # for i in range(dist.world_size):
#     #     if i == dist.rank:
#     #         print(f'[[[[ rk {dist.rank} ]]]]: dist.dev_idx={dist.dev_idx}, gpu_dev_idx={gpu_dev_idx}')
#     #     dist.barrier()
#     # assert dist.dev_idx == gpu_dev_idx
#
#     args.descs = [f'rk{rk:02d}' for rk in range(dist.world_size)]
#     args.num_classes = {
#         'cifar10': 10,
#         'cifar100': 100,
#         'imagenet': 1000,
#     }[args.dataset]
#     # todo: change desc when doing a grid search
#
#     args.loc_desc = args.descs[dist.rank]
#     lg, g_tb_lg, l_tb_lg = create_files(args, dist)
#     lg: Logger = lg  # just for the code completion (actually is `DistLogger`)
#     g_tb_lg: SummaryWriter = g_tb_lg  # just for the code completion (actually is `DistLogger`)
#     l_tb_lg: SummaryWriter = l_tb_lg  # just for the code completion (actually is `DistLogger`)
#     lg.info(f'=> [main]: args:\n{pf(vars(args))}\n')
#
#     if args.seed_base is None:
#         lg.info(f'=> [main]: args.seed_base is None, no set_seed called')
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False
#     else:
#         seeds = torch.zeros(dist.world_size).float()
#         seeds[dist.rank] = args.seed = args.seed_base + dist.rank
#         dist.allreduce(seeds)
#         dist.broadcast(seeds, 0)
#         assert torch.allclose(seeds, torch.arange(args.seed_base, args.seed_base + dist.world_size).float())
#         same_seed = args.torch_ddp
#         set_seed(args.seed_base if same_seed else args.seed)
#         lg.info(f'=> [main]: using {"the same seed" if same_seed else "diff seeds"}')
#
#     upd_seatable_file(
#         args.exp_root, dist,
#         gpu=dist.world_size if args.torch_ddp else 1,
#         ds=args.dataset,
#         # mom=args.moco_m,
#         T=args.moco_t,
#         sbn=args.sbn, mlp=args.mlp, sym=args.moco_symm, init=args.init,
#         ep=args.epochs, bs=args.batch_size, cos=args.coslr, wp=args.warmup, nowd=args.nowd,
#         v_ep=args.eval_epochs, v_bs=args.eval_batch_size, v_cos=args.eval_coslr, v_wp=args.eval_warmup, v_nowd=args.eval_nowd,
#         pr=0, rem=0, beg_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#     )
#
#     def get_normalize(ds_name: str):
#         return transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#
#     assert args.dataset == 'cifar10'
#     pret_transform = transforms.Compose([
#         transforms.RandomResizedCrop(32),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         get_normalize(args.dataset)
#     ])
#
#     # todo: here
#     candidates_geo = [
#         ShearX(ShearX.RANGES[4]), ShearY(ShearY.RANGES[4]),
#         TranslateX(TranslateX.RANGES[4]), TranslateY(TranslateY.RANGES[4]),
#         Rotate(Rotate.RANGES[4]),
#     ]
#     candidates_col = [
#         Color(Color.RANGES[4]), Contrast(Contrast.RANGES[4]), Brightness(Brightness.RANGES[4]),
#         Sharpness(Sharpness.RANGES[4]),
#         AutoContrast(), Invert(),
#
#     ]
#     swap_transform = transforms.Compose([
#         transforms.RandomResizedCrop(32),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomApply([AutoContrast()], p=0.5),
#         transforms.RandomApply([Sharpness(Sharpness.RANGES[4])], p=0.5),
#         transforms.ToTensor(),
#         get_normalize(args.dataset)
#     ])
#
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         get_normalize(args.dataset)
#     ])
#     eval_transform = transforms.Compose([
#         transforms.RandomResizedCrop(32),
#         transforms.RandomHorizontalFlip(p=0.5),
#         # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         # transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         get_normalize(args.dataset)
#     ])
#
#     ds_root = args.ds_root or os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', args.dataset))
#
#     assert not args.torch_ddp
#     data_kw = dict(num_workers=args.num_workers, pin_memory=args.pin_mem)
#
#     pret_data = CIFAR10Pair(root=ds_root, train=True, transform=pret_transform, download=False)
#     pret_loader = DataLoader(pret_data, args.batch_size, drop_last=True, shuffle=True, **data_kw)
#     pret_iters = len(pret_loader)
#     lg.info(f'=> [main]: prepare pret_data (iters={pret_iters}, ddp={args.torch_ddp}): @ {args.dataset}')
#
#     swap_data = CIFAR10Pair(root=ds_root, train=True, transform=swap_transform, download=False)
#     swap_loader = DataLoader(swap_data, args.batch_size, drop_last=True, shuffle=True, **data_kw)
#     swap_iters = len(swap_loader)
#     lg.info(f'=> [main]: prepare swap_data (iters={swap_iters}, ddp={args.torch_ddp}): @ {args.dataset}')
#
#     knn_data = CIFAR10(root=ds_root, train=True, transform=test_transform, download=False)
#     knn_loader = DataLoader(knn_data, args.batch_size * 2, drop_last=False, shuffle=False, **data_kw)
#     knn_iters = len(knn_loader)
#     lg.info(f'=> [main]: prepare knn_data (iters={knn_iters}, ddp={args.torch_ddp}): @ {args.dataset}')
#
#     test_data = CIFAR10(root=ds_root, train=False, transform=test_transform, download=False)
#     test_loader = DataLoader(test_data, args.batch_size * 2, drop_last=False, shuffle=False, **data_kw)
#     test_iters = len(test_loader)
#     lg.info(f'=> [main]: prepare test_data (iters={test_iters}, ddp={args.torch_ddp}): @ {args.dataset}')
#
#     eval_data = CIFAR10(root=ds_root, train=True, transform=eval_transform, download=False)
#     eval_loader = DataLoader(eval_data, args.batch_size, drop_last=True, shuffle=True, **data_kw)
#     eval_iters = len(eval_loader)
#     lg.info(f'=> [main]: prepare eval_data (iters={eval_iters}, ddp={args.torch_ddp}): @ {args.dataset}')
#
#     lg.info(
#         f'=> [main]: create the moco model: (ddp={args.torch_ddp})\n'
#         f'     arch={args.arch}, feature dim={args.moco_dim}\n'
#         f'     Q size={args.moco_k}, ema mom={args.moco_m}, moco T={args.moco_t}\n'
#         f'     sync bn={args.sbn}, mlp={args.mlp}, symmetric loss={args.moco_symm}'
#     )
#     # create model
#     pretrain_model = ModelMoCo(
#         lg=lg,
#         torch_ddp=args.torch_ddp,
#         arch=args.arch,
#         dim=args.moco_dim,
#         K=args.moco_k,  # queue size
#         m=args.moco_m,  # ema momentum
#         T=args.moco_t,  # temperature
#         sbn=args.sbn,
#         mlp=args.mlp,
#         symmetric=args.moco_symm,
#         init=args.init
#     ).cuda()
#
#     lnr_eval_model = ModelMoCo(
#         lg=lg,
#         torch_ddp=args.torch_ddp,
#         arch=args.arch,
#         dim=args.num_classes,
#         K=args.moco_k,  # queue size
#         m=args.moco_m,  # ema momentum
#         T=args.moco_t,  # temperature
#         sbn=args.sbn,
#         mlp=args.mlp,
#         symmetric=args.moco_symm,
#         init=args.init
#     ).cuda()
#
#     if args.eval_resume_ckpt is None:
#         pretrain_or_linear_eval(
#             (knn_iters, knn_loader, args.knn_k, args.knn_t), args.num_classes, ExpMeta(
#                 args.torch_ddp, args.arch, args.exp_root, args.exp_dirname, args.descs, args.log_freq, args.resume_ckpt,
#                 args.epochs, args.lr, args.wd, args.nowd, args.coslr, args.schedule, args.warmup, args.grad_clip
#             ),
#             lg, g_tb_lg, l_tb_lg, dist, pretrain_model, pret_iters, pret_loader, test_iters, test_loader
#         )
#         d = pretrain_model.encoder_q.state_dict()
#         ks = deepcopy(list(d.keys()))
#         for k in ks:
#             if k.startswith('fc.'):
#                 del d[k]
#         msg = lnr_eval_model.encoder_q.load_state_dict(d, strict=False)
#         assert len(msg.unexpected_keys) == 0 and all(k.startswith('fc.') for k in msg.missing_keys)
#
#     pretrain_or_linear_eval(
#         None, args.num_classes, ExpMeta(
#             args.torch_ddp, args.arch, args.exp_root, args.exp_dirname, args.descs, args.log_freq, args.eval_resume_ckpt,
#             args.eval_epochs, args.eval_lr, args.eval_wd, args.eval_nowd, args.eval_coslr, args.eval_schedule, args.eval_warmup, args.eval_grad_clip
#         ),
#         lg, g_tb_lg, l_tb_lg, dist, lnr_eval_model.encoder_q, eval_iters, eval_loader, test_iters, test_loader
#     )
#
#     g_tb_lg.close()
#     l_tb_lg.close()
#     # dist.finalize()
#
#
# class ExpMeta(NamedTuple):
#     # configs
#     torch_ddp: bool
#     arch: str
#     exp_root: str
#     exp_dirname: str
#     descs: List[str]
#     log_freq: int
#     resume_ckpt: Optional[str]
#
#     # hyperparameters
#     epochs: int
#     lr: float
#     wd: float
#     nowd: bool
#     coslr: bool
#     schedule: List[int]
#     warmup: bool
#     grad_clip: float
#
#
# def pretrain_or_linear_eval(
#         pretrain_knn_args, num_classes: int,
#         meta: ExpMeta, lg: Logger, g_tb_lg: SummaryWriter, l_tb_lg: SummaryWriter,
#         dist: TorchDistManager, model: Union[ModelMoCo, torch.nn.Module],
#         tr_iters: int, tr_loader: DataLoader, te_iters: int, te_loader: DataLoader,
#
# ):
#     is_pretrain = pretrain_knn_args is not None
#     prefix = 'pretrain' if is_pretrain else 'lnr_eval'
#     test_acc_name = 'knn_acc' if is_pretrain else 'test_acc'
#     assert is_pretrain == isinstance(model, ModelMoCo)
#
#     if not meta.coslr:
#         sc = meta.schedule
#         lg.info(f'=> [{prefix}]: origin lr schedule={sc} ({type(sc)})')
#         if isinstance(sc, str):
#             sc = eval(sc)
#             assert isinstance(sc, list)
#         sc = sorted(sc)
#         for i, milestone_epoch in enumerate(sc):
#             sc[i] = milestone_epoch * tr_iters
#         lg.info(f'=> [{prefix}]: updated lr schedule={sc} ({type(sc)})')
#         meta = meta._replace(schedule=sc)
#
#     # load model if resume
#     epoch_start = 0
#     best_test_acc1 = 0
#     best_test_acc5 = 0
#     topk_acc1s = TopKHeap(maxsize=max(1, round(meta.epochs * 0.05)))
#     # todo: double-check ckpt loading and early returning
#     if meta.resume_ckpt is not None:
#         ckpt = torch.load(meta.resume_ckpt, map_location='cpu')
#         model.load_state_dict(ckpt['model'])
#         best_test_acc1 = ckpt['best_test_acc1']
#         [topk_acc1s.push_q(x) for x in ckpt['topk_acc1s']]
#
#         epoch_start = ckpt['epoch'] + 1
#         lg.info(f'=> [{prefix}]: ckpt loaded from {meta.resume_ckpt}, last_ep={epoch_start - 1}, ep_start={epoch_start}')
#         if epoch_start == meta.epochs:
#             return
#     else:
#         ckpt = None
#
#     if not is_pretrain:
#         initial_model_state = deepcopy(model.state_dict())
#         for p in model.parameters():
#             p.detach_()
#         for m in model.fc.modules():
#             clz_name = m.__class__.__name__
#             if 'Linear' in clz_name:
#                 m.weight.data.normal_(mean=0.0, std=0.01)
#                 m.weight.requires_grad_()
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                     m.bias.requires_grad_()
#     else:
#         initial_model_state = None
#
#     params = filter_params(model) if meta.nowd else model.parameters()
#     params = list(filter(lambda p: p.requires_grad, params))
#     optimizer = torch.optim.SGD(params, lr=meta.lr, weight_decay=meta.wd, momentum=0.9)
#     lg.info(f'=> [{prefix}]: create op: model_cls={model.__class__.__name__}, len(params)={len(params)}, max_lr={meta.lr}, wd={meta.wd}, nowd={meta.nowd}, coslr={meta.coslr}, warm up={meta.warmup}')
#     if ckpt is not None:
#         lg.info(f'=> [{prefix}]: load optimizer.state from {meta.resume_ckpt}')
#         optimizer.load_state_dict(ckpt['optimizer'])
#
#     loop_start_t = time.time()
#     epoch_speed = AverageMeter(3)
#     tr_loss_avg = AverageMeter(tr_iters)
#     tr_acc1_avg = AverageMeter(tr_iters)
#     tr_acc5_avg = AverageMeter(tr_iters)
#     avgs = (tr_loss_avg, tr_acc1_avg, tr_acc5_avg)
#     for epoch in range(epoch_start, meta.epochs):
#         ep_str = f'%{len(str(meta.epochs))}d'
#         ep_str %= epoch + 1
#         if epoch % 5 == 0 and dist.is_master():
#             em_t = time.time()
#             torch.cuda.empty_cache()
#             os.system(f'echo -e "\033[36m @@@@@ {meta.exp_root} , ept_cc: {time.time() - em_t:.3f}s \033[0m"')
#
#         start_t = time.time()
#         tr_loss = train(is_pretrain, prefix, lg, g_tb_lg, l_tb_lg, dist, meta, epoch, ep_str, tr_iters, tr_loader, model, params, optimizer, avgs)
#         train_t = time.time()
#
#         if is_pretrain:
#             test_acc1 = knn_test(lg, l_tb_lg, dist, meta.log_freq, epoch, ep_str, te_iters, te_loader, model.encoder_q, pretrain_knn_args, num_classes)
#             test_acc5, test_loss = 0, 0
#         else:
#             test_acc1, test_acc5, test_loss = eval_test(lg, l_tb_lg, dist, meta.log_freq, epoch, ep_str, te_iters, te_loader, model, num_classes)
#             l_tb_lg.add_scalar(f'{prefix}/{test_acc_name}5', test_acc5, epoch + 1)
#             l_tb_lg.add_scalar(f'{prefix}/{test_acc_name.replace("acc", "loss")}', test_loss, epoch + 1)
#         l_tb_lg.add_scalar(f'{prefix}/{test_acc_name}1', test_acc1, epoch + 1)
#         test_t = time.time()
#
#         topk_acc1s.push_q(test_acc1)
#         best_updated = best_test_acc1 < test_acc1
#         state_dict = {
#             'arch': meta.arch, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
#             'topk_acc1s': list(topk_acc1s),
#         }
#         if best_updated:
#             best_test_acc1 = test_acc1
#             best_test_acc5 = test_acc5
#             torch.save(state_dict, os.path.join(meta.exp_root, f'{prefix}_best.pth'))
#         torch.save(state_dict, os.path.join(meta.exp_root, f'{prefix}_latest.pth'))
#
#         remain_time, finish_time = epoch_speed.time_preds(meta.epochs - (epoch + 1))
#         lg.info(
#             f'=> [ep {ep_str}/{meta.epochs}]: L={tr_loss:.2g}, te-acc={test_acc1:5.2f}, tr={train_t - start_t:.2f}s, te={test_t - train_t:.2f}s       best={best_test_acc1:5.2f}\n'
#             f'    {prefix} [{str(remain_time)}] ({finish_time})'
#         )
#         if dist.is_master():
#             lr_str = f'{optimizer.param_groups[0]["lr"] / meta.lr:.1e}'
#             kw = {test_acc_name: test_acc1, 'lr' if is_pretrain else 'v_lr': lr_str}
#             upd_seatable_file(
#                 meta.exp_root, dist, pr=(epoch + 1) / meta.epochs,
#                 rem=remain_time.seconds, end_t=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_time.seconds)),
#                 **kw
#             )
#
#         epoch_speed.update(time.time() - start_t)
#         if epoch == epoch_start:
#             print(f'[rk{dist.rank:2d}] barrier test')
#             dist.barrier()
#             if not is_pretrain:
#                 sanity_check(model.state_dict(), initial_model_state)
#
#     topk_test_acc1 = sum(topk_acc1s) / len(topk_acc1s)
#     dt = time.time() - loop_start_t
#     if not meta.torch_ddp:
#         topk_accs = dist.dist_fmt_vals(topk_test_acc1, None)
#         best_accs = dist.dist_fmt_vals(best_test_acc1, None)
#         best_acc5s = dist.dist_fmt_vals(best_test_acc5, None)
#         [g_tb_lg.add_scalar(f'{prefix}/{test_acc_name.replace("acc", "best")}1', best_accs.mean().item(), e) for e in [epoch_start, meta.epochs]]
#         perform_dict = pf({
#             des: f'topk={ta.item():.3f}, best={ba.item():.3f}'
#             for des, ta, ba in zip(meta.descs, topk_accs, best_accs)
#         })
#         res_str = (
#             f' best     acc5s @ (max={best_acc5s.max():.3f}, mean={best_acc5s.mean():.3f}, std={best_acc5s.std():.3f}) {str(best_acc5s).replace(chr(10), " ")})\n'
#             f' mean-top acc1s @ (max={topk_accs.max():.3f}, mean={topk_accs.mean():.3f}, std={topk_accs.std():.3f}) {str(topk_accs).replace(chr(10), " ")})\n'
#             f' best     acc1s @ (max={best_accs.max():.3f}, mean={best_accs.mean():.3f}, std={best_accs.std():.3f}) {str(best_accs).replace(chr(10), " ")})'
#         )
#         lg.info(
#             f'==> {prefix} finished,'
#             f' total time cost: {dt / 60:.2f}min ({dt / 60 / 60:.2f}h)'
#             f' topk: {pf([round(x, 2) for x in topk_acc1s])}\n'
#             f' performance: \n{perform_dict}\n{res_str}'
#         )
#
#         if dist.is_master():
#             kw = {test_acc_name: best_accs.mean().item(), 'lr' if is_pretrain else 'v_lr': f'{meta.lr:.1g}'}
#             upd_seatable_file(
#                 meta.exp_root, dist, pr=1., rem=0,
#                 end_t=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 **kw
#             )
#             lines = res_str.splitlines()
#             strs = ''.join([f'# {l}\n' for l in lines])
#             with open(run_shell_name, 'a') as fp:
#                 print(f'\n# {prefix} {meta.exp_dirname}:\n{strs}', file=fp)
#
#         if is_pretrain:  # sync parameters
#             src_rank = topk_accs.argmax().item()
#             for _, param in model.state_dict().items():
#                 dist.broadcast(param.data, src_rank)
#
#     else:
#         assert False
#
#
# # pretrain for one epoch
# def train(is_pretrain, prefix, lg, g_tb_lg, l_tb_lg, dist, meta: ExpMeta, epoch, ep_str, tr_iters, tr_loader, model, params, op, avgs):
#     if is_pretrain:
#         model.train()
#     else:
#         model.eval()
#
#     tr_loss_avg, tr_acc1_avg, tr_acc5_avg = avgs
#     log_iters = tr_iters // meta.log_freq
#
#     tot_loss, tot_num = 0.0, 0
#     last_t = time.time()
#     for it, (data1, data2) in enumerate(tr_loader):
#         data_t = time.time()
#         bs = data1.shape[0]
#         cur_iter = it + epoch * tr_iters
#         max_iter = meta.epochs * tr_iters
#         data1, data2 = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True)
#         cuda_t = time.time()
#
#         if is_pretrain:
#             loss = model(data1, data2)
#         else:
#             oup = model(data1)
#             loss = F.cross_entropy(oup, data2)
#             acc1, acc5 = accuracy(oup, data2, topk=(1, 5))
#             tr_acc1_avg.update(acc1, bs)
#             tr_acc5_avg.update(acc5, bs)
#         l = loss.item()
#         tr_loss_avg.update(l, bs)
#         tot_num += bs
#         tot_loss += l
#         forw_t = time.time()
#
#         op.zero_grad()
#         loss.backward()
#         back_t = time.time()
#         sche_lr = adjust_learning_rate(op, cur_iter, max_iter, meta.lr, meta)
#         clipping = cur_iter < tr_iters * 5
#         if clipping:
#             orig_norm = torch.nn.utils.clip_grad_norm_(params, meta.grad_clip)
#         else:
#             orig_norm = meta.grad_clip
#         actual_lr = sche_lr * min(1, meta.grad_clip / orig_norm)
#         clip_t = time.time()
#
#         op.step()
#         step_t = time.time()
#
#         if cur_iter < 10 or cur_iter % log_iters == 0 or (actual_lr < sche_lr - 1e-6 and random.randrange(8) == 0):
#             g_tb_lg.add_scalar(f'{prefix}/orig_norm', orig_norm, cur_iter)
#             g_tb_lg.add_scalars(f'{prefix}/lr', {'scheduled': sche_lr}, cur_iter)
#             if clipping:
#                 g_tb_lg.add_scalars(f'{prefix}/lr', {'actual': actual_lr}, cur_iter)
#
#         if cur_iter % log_iters == 0:
#             # l_tb_lg.add_scalars(f'{prefix}/tr_loss', {'it': loss_avg.avg}, cur_iter)
#             l_tb_lg.add_scalar(f'{prefix}/train_loss', tr_loss_avg.avg, cur_iter)
#             if is_pretrain:
#                 acc_str = ''
#             else:
#                 acc_str = f'tr_a1={tr_acc1_avg.avg:5.2f}, tr_a5={tr_acc5_avg.avg:5.2f}'
#                 l_tb_lg.add_scalar(f'{prefix}/train_acc1', tr_acc1_avg.avg, cur_iter)
#                 l_tb_lg.add_scalar(f'{prefix}/train_acc5', tr_acc5_avg.avg, cur_iter)
#             lg.info(
#                 f'\n'
#                 f'    ep[{ep_str}] it[{it + 1}/{tr_iters}]: L={tr_loss_avg.avg:.2g} {acc_str}\n'
#                 f'     {prefix} da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fo[{forw_t - cuda_t:.3f}], ba[{back_t - forw_t:.3f}], '
#                 f'cl[{clip_t-back_t:.3f}], op[{step_t-clip_t:.3f}]'
#             )
#
#         last_t = time.time()
#
#     return tot_loss / tot_num
#
#
# # test using a knn monitor
# def knn_test(lg, l_tb_lg, dist, log_freq, epoch, ep_str, te_iters, te_loader, moco_encoder_q, knn_args, num_classes):
#     knn_iters, knn_loader, knn_k, knn_t = knn_args
#     log_iters = te_iters // log_freq
#
#     moco_encoder_q.eval()
#     total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
#     with torch.no_grad():
#         # generate feature bank
#         for it, (inp, tar) in enumerate(knn_loader):
#             feature = moco_encoder_q(inp.cuda(non_blocking=True))
#             feature = F.normalize(feature, dim=1)
#             feature_bank.append(feature)
#         # [D, N]
#         feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
#         # [N]
#         feature_labels = torch.tensor(knn_loader.dataset.targets, device=feature_bank.device)
#
#         # loop test data to predict the label by weighted knn search
#         last_t = time.time()
#         for it, (inp, tar) in enumerate(te_loader):
#             data_t = time.time()
#             inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
#             cuda_t = time.time()
#             feature = moco_encoder_q(inp)
#             feature = F.normalize(feature, dim=1)
#             fea_t = time.time()
#
#             pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, knn_k, knn_t)
#             pred_t = time.time()
#
#             total_num += inp.shape[0]
#             total_top1 += (pred_labels[:, 0] == tar).float().sum().item()
#
#             if it % log_iters == 0:
#                 cur_te_acc1 = total_top1 / total_num * 100
#                 lg.info(
#                     f'     ep[{ep_str}] it[{it + 1}/{te_iters}]: *test acc={cur_te_acc1:5.3f}\n'
#                     f'       da[{data_t - last_t:.3f}], cu[{cuda_t - data_t:.3f}], fe[{fea_t - cuda_t:.3f}], kn[{pred_t - fea_t:.3f}]'
#                 )
#
#             last_t = time.time()
#
#     return total_top1 / total_num * 100
#
#
# # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
# def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
#     # compute cos similarity between each feature vector and feature bank ---> [B, N]
#     sim_matrix = torch.mm(feature, feature_bank)
#     # [B, K]
#     sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
#     # [B, K]
#     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
#     sim_weight = (sim_weight / knn_t).exp()
#
#     # counts for each class
#     one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
#     # [B*K, C]
#     one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#     # weighted score ---> [B, C]
#     pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
#
#     pred_labels = pred_scores.argsort(dim=-1, descending=True)
#     return pred_labels
#
#
# def eval_test(lg, l_tb_lg, dist, log_freq, epoch, ep_str, te_iters, te_loader, moco_encoder_q, num_classes):
#     moco_encoder_q.eval()
#     with torch.no_grad():
#         tot_acc1, tot_acc5, tot_loss, tot_num = 0, 0, 0, 0
#         for it, (inp, tar) in enumerate(te_loader):
#             bs = inp.shape[0]
#             inp, tar = inp.cuda(non_blocking=True), tar.cuda(non_blocking=True)
#             oup = moco_encoder_q(inp)
#             assert oup.shape[1] == num_classes
#             loss = F.cross_entropy(oup, tar)
#             acc1, acc5 = accuracy(oup, tar, topk=(1, 5))
#             tot_acc1 += acc1 * bs
#             tot_acc5 += acc5 * bs
#             tot_loss += loss.item() * bs
#             tot_num += bs
#
#         return tot_acc1 / tot_num, tot_acc5 / tot_num, tot_loss / tot_num
#
#
# def sanity_check(current_state, initial_state):
#     ks1, ks2 = list(current_state.keys()), list(initial_state.keys())
#     assert ks1 == ks2
#     for key in filter(lambda k: not k.startswith('fc.'), ks1):
#         t1, t2 = current_state[key], initial_state[key]
#         t1: torch.Tensor
#         t2: torch.Tensor
#         assert (t1 == t2).all(), f'{key} changed in the linear evaluation'
#
#
# if __name__ == '__main__':
#     main()
