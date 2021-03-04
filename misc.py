import torch


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
            torch.nn.init.normal_(m.weight, std=1/m.weight.size(-1))
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
