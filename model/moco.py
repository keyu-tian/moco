from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F

from model import model_entry, resnet18
from model.bn import SplitBatchNorm
from utils.misc import init_params


class ModelMoCo(nn.Module):
    def __init__(self, lg, on_imagenet, torch_ddp=False, arch='resnet18', dim=128, K=4096, m=0.99, T=0.1, sbn=False, mlp=False, symmetric=True, init=False):
        super(ModelMoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.torch_ddp = torch_ddp
        
        # create the encoders
        if on_imagenet:
            if sbn:
                pass    # todo: 怎么办，现在gb256 8卡会炸1080ti；现在已经把b256改成b128了；moco_k从65536改成2048。已经64了！
            norm_layer = nn.BatchNorm2d
        else:
            bn_splits = 1 if sbn else 8
            norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        
        self.encoder_q = model_entry(model_name=arch, num_classes=dim, norm_layer=norm_layer)
        self.encoder_k = model_entry(model_name=arch, num_classes=dim, norm_layer=norm_layer)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        
        if init:
            init_params(self.encoder_q, output=lg.info)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.detach_()                   # not update by gradient
        
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
        if self.torch_ddp:
            keys = concat_all_gather(keys)
        
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

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
    
        num_gpus = batch_size_all // batch_size_this
    
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
    
        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
    
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
    
        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
    
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
    
        num_gpus = batch_size_all // batch_size_this
    
        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
    
        return x_gather[idx_this]
    
    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            if self.torch_ddp:
                im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            else:
                im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
            
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            
            # undo shuffle
            if self.torch_ddp:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
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
        
        loss = F.cross_entropy(logits, labels)
        
        return loss, q, k
    
    def forward(self, im1, im2, training=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        
        # update the key encoder
        if training:
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
        
        if training:
            self._dequeue_and_enqueue(k)
        
        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    m = ModelMoCo(None, mlp=True)
    q = m.encoder_q
    for name, param in q.named_parameters():
        # print(name, param)
        break
    r = resnet18(num_classes=128)
    kk = list(q.state_dict().keys())
    d = q.state_dict()
    for k in kk:
        if k.startswith('fc.'):
            del d[k]
    
    msg = q.load_state_dict(d, strict=False)
    print(msg)

    assert len(msg.unexpected_keys) == 0 and all(k.startswith('fc.') for k in msg.missing_keys)
    



