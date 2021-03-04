# #!/usr/bin/env python
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import argparse
# import builtins
# import math
# import os
# import random
# import shutil
# import time
# import warnings
#
# import torch
# import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim
# import torch.multiprocessing as mp
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# from easydict import EasyDict
#
# import moco.loader
# import moco.builder
#
#
# args = EasyDict()
# args.data = '/path/to/ds'
# args.arch = 'resnet50'
# args.workers = 32
# args.epochs = 200
# args.start_epoch = 0
# args.batch_size = 256
#
# args.lr = 0.03
# args.cos = True
# args.schedule = [120, 160]
# args.wd = 1e-4
# args.momentum = 0.9
#
# args.print_freq = 10
# args.resume = ''
# args.dist_backend = 'nccl'
# args.dist_url = 'tcp://localhost:10001'
# args.world_size = 1
# args.rank = 0
# args.seed: int = None
# args.gpu: int = None
# args.multiprocessing_distributed = True
# args.distributed = args.world_size > 1 or args.multiprocessing_distributed
#
# args.mlp = True
# args.aug_plus = True
# args.moco_dim = 128
# args.moco_k = 65536
# args.moco_m = 0.999
# args.moco_t = 0.2
#
#
# def main():
#
#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')
#
#     if args.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')
#
#     if args.dist_url == "env://" and args.world_size == -1:
#         args.world_size = int(os.environ["WORLD_SIZE"])
#
#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed
#
#     ngpus_per_node = torch.cuda.device_count()
#     if args.multiprocessing_distributed:
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         args.world_size = ngpus_per_node * args.world_size
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
#     else:
#         # Simply call main_worker function
#         main_worker(args.gpu, ngpus_per_node)
#
#
# def main_worker(gpu, ngpus_per_node):
#     args.gpu = gpu
#
#     # suppress printing if not master
#     if args.multiprocessing_distributed and args.gpu != 0:
#         def print_pass(*args):
#             pass
#         builtins.print = print_pass
#
#     if args.gpu is not None:
#         print("Use GPU: {} for training".format(args.gpu))
#
#     if args.distributed:
#         if args.dist_url == "env://" and args.rank == -1:
#             args.rank = int(os.environ["RANK"])
#         if args.multiprocessing_distributed:
#             # For multiprocessing distributed training, rank needs to be the
#             # global rank among all the processes
#             args.rank = args.rank * ngpus_per_node + gpu
#         dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                 world_size=args.world_size, rank=args.rank)
#     # create model
#     print("=> creating model '{}'".format(args.arch))
#     model = moco.builder.MoCo(
#         models.__dict__[args.arch],
#         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
#     print(model)
#
#     if args.distributed:
#         # For multiprocessing distributed, DistributedDataParallel constructor
#         # should always set the single device scope, otherwise,
#         # DistributedDataParallel will use all available devices.
#         if args.gpu is not None:
#             torch.cuda.set_device(args.gpu)
#             model.cuda(args.gpu)
#             # When using a single GPU per process and per
#             # DistributedDataParallel, we need to divide the batch size
#             # ourselves based on the total number of GPUs we have
#             args.batch_size = int(args.batch_size / ngpus_per_node)
#             args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#         else:
#             model.cuda()
#             # DistributedDataParallel will divide and allocate batch_size to all
#             # available GPUs if device_ids are not set
#             model = torch.nn.parallel.DistributedDataParallel(model)
#     elif args.gpu is not None:
#         torch.cuda.set_device(args.gpu)
#         model = model.cuda(args.gpu)
#         # comment out the following line for debugging
#         raise NotImplementedError("Only DistributedDataParallel is supported.")
#     else:
#         # AllGather implementation (batch shuffle, queue update, etc.) in
#         # this code only supports DistributedDataParallel.
#         raise NotImplementedError("Only DistributedDataParallel is supported.")
#
#     # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda(args.gpu)
#
#     optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
#
#     # optionally resume from a checkpoint
#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))
#             if args.gpu is None:
#                 checkpoint = torch.load(args.resume)
#             else:
#                 # Map model to be loaded to specified single gpu.
#                 loc = 'cuda:{}'.format(args.gpu)
#                 checkpoint = torch.load(args.resume, map_location=loc)
#             args.start_epoch = checkpoint['epoch']
#             model.load_state_dict(checkpoint['state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resume))
#
#     cudnn.benchmark = True
#
#     # Data loading code
#     traindir = os.path.join(args.data, 'train')
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     if args.aug_plus:
#         # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
#         augmentation = [
#             transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ]
#     else:
#         # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
#         augmentation = [
#             transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ]
#
#     train_dataset = datasets.ImageFolder(
#         traindir,
#         moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
#
#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
#
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             train_sampler.set_epoch(epoch)
#         adjust_learning_rate(optimizer, epoch)
#
#         # train for one epoch
#         train(train_loader, model, criterion, optimizer, epoch)
#
#         if not args.multiprocessing_distributed or (args.multiprocessing_distributed
#                 and args.rank % ngpus_per_node == 0):
#             save_checkpoint({
#                 'epoch': epoch + 1,
#                 'arch': args.arch,
#                 'state_dict': model.state_dict(),
#                 'optimizer' : optimizer.state_dict(),
#             }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
#
#
# def train(train_loader, model, criterion, optimizer, epoch):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (images, _) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if args.gpu is not None:
#             images[0] = images[0].cuda(args.gpu, non_blocking=True)
#             images[1] = images[1].cuda(args.gpu, non_blocking=True)
#
#         # compute output
#         output, target = model(im_q=images[0], im_k=images[1])
#         loss = criterion(output, target)
#
#         # acc1/acc5 are (K+1)-way contrast classifier accuracy
#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), images[0].size(0))
#         top1.update(acc1[0], images[0].size(0))
#         top5.update(acc5[0], images[0].size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             progress.display(i)
#
#
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)
#
#
# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix
#
#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))
#
#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'
#
#
# def adjust_learning_rate(optimizer, epoch):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if args.cos:  # cosine lr schedule
#         lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
#     else:  # stepwise lr schedule
#         for milestone in args.schedule:
#             lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#
# if __name__ == '__main__':
#     main()
