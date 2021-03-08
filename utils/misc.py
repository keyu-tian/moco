import datetime
import heapq
import os
import random
import socket
import time
from collections import defaultdict

import numpy as np
import torch


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def ints_ceil(x: int, y: int) -> int:
    return (x + y - 1) // y  # or (x - 1) // y + 1


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MaxHeap(list):
    
    def __init__(self, maxsize):
        super(MaxHeap, self).__init__()
        self.maxsize = maxsize
        assert self.maxsize > 1
    
    def push_q(self, x):
        if len(self) < self.maxsize:
            heapq.heappush(self, x)
        elif x > self[0]:
            heapq.heappushpop(self, x)
    
    def pop_q(self):
        return heapq.heappop(self)
    
    def __repr__(self):
        return str(sorted([x for x in self], reverse=True))


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = round(length)
        if self.length > 0:
            self.queuing = True
            self.val_history = []
            self.num_history = []
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def reset(self):
        if self.length > 0:
            self.val_history.clear()
            self.num_history.clear()
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def update(self, val, num=1):
        self.val_sum += val * num
        self.num_sum += num
        self.last = val
        if self.queuing:
            self.val_history.append(val)
            self.num_history.append(num)
            if len(self.val_history) > self.length:
                self.val_sum -= self.val_history[0] * self.num_history[0]
                self.num_sum -= self.num_history[0]
                del self.val_history[0]
                del self.num_history[0]
        self.avg = self.val_sum / self.num_sum
    
    def time_preds(self, counts):
        remain_secs = counts * self.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        return remain_time, finish_time
    
    def state_dict(self):
        return vars(self)
    
    def load_state(self, state_dict):
        self.__dict__.update(state_dict)
    

def init_params(model: torch.nn.Module, output=None):
    if output is not None:
        output('===================== param initialization =====================')
    tot_num_inited = 0
    for i, m in enumerate(model.modules()):
        clz = m.__class__.__name__
        is_conv = clz.find('Conv') != -1
        is_bn = clz.find('BatchNorm') != -1
        is_fc = clz.find('Linear') != -1
        
        cur_num_inited = []
        if is_conv:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_bn:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_fc:
            # torch.nn.init.normal_(m.weight, std=0.001)
            torch.nn.init.normal_(m.weight, std=1 / m.weight.size(-1))
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        tot_num_inited += sum(cur_num_inited)
        
        if output is not None:
            builtin = any((is_conv, is_bn, is_fc))
            cur_num_inited = f' ({" + ".join([str(x) for x in cur_num_inited])})'
            output(f'clz{i:3d}: {"  => " if builtin else ""}{clz}{cur_num_inited if builtin else "*"}')
    
    if output is not None:
        output('----------------------------------------------------------------')
        output(f'tot_num_inited: {tot_num_inited} ({tot_num_inited / 1e6:.3f} M)')
        output('===================== param initialization =====================\n')
    return tot_num_inited


def filter_params(model: torch.nn.Module):
    special_decay_rules = {
        'bn_b': {'weight_decay': 0.0},
        'bn_w': {'weight_decay': 0.0},
    }
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in special_decay_rules:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in special_decay_rules:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in special_decay_rules:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in special_decay_rules:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in special_decay_rules:
        pgroup['linear_w'] = []
        names['linear_w'] = []
    
    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        clz = m.__class__.__name__
        if clz.find('Conv') != -1:
            if m.bias is not None:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dw_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dense_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dw_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dense_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dense)'] += 1
        
        elif clz.find('Linear') != -1:
            if m.bias is not None:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['linear_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['linear_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
        
        elif clz.find('BatchNorm') != -1:
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['bn_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['bn_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
    
    for name, p in model.named_parameters():
        if name not in names_all:
            pgroup_normal.append(p)
    
    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in special_decay_rules.keys():
            param_groups.append({'params': pgroup[ptype], **special_decay_rules[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})
        
        # if logger is not None:
        #     logger.info(ptype)
        #     for k, v in param_groups[-1].items():
        #         if k == 'params':
        #             logger.info('   params: {}'.format(len(v)))
        #         else:
        #             logger.info('   {}: {}'.format(k, v))
    
    # if logger is not None:
    #     for ptype, pconf in special_decay.items():
    #         logger.info('names for {}({}): {}'.format(ptype, len(names[ptype]), names[ptype]))
    
    # return param_groups, type2num
    return param_groups
