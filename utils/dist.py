import json
import os
import socket
import time
from typing import List, Union
from colorama import Fore

import torch
import torch.distributed as dist
from torch.multiprocessing import set_start_method

from utils.misc import time_str


class TorchDistManager:
    WORLD_GROUP = dist.group.WORLD if torch.cuda.is_available() else None
    
    def __init__(self, node0_addr: Union[int, str], node0_port: Union[int, str], mp_start_method: str = 'fork', backend: str = 'nccl'):
        set_start_method(mp_start_method, force=True)
        self.backend = backend
        # if multi_nodes:   # if $2 > ntasks-per-node
        #     os.environ[f'{backend.upper()}_SOCKET_IFNAME'] = 'eth0'
        world_size: int = int(os.environ['SLURM_NTASKS'])
        rank: int = int(os.environ['SLURM_PROCID'])
        
        temp_f_path = os.path.join(os.getcwd(), f'.temp_{os.environ["SLURM_NODELIST"]}.json')
        self.temp_f_path = temp_f_path
        if rank == 0:
            node0_addr = str(node0_addr).lower()
            if node0_addr == 'auto':
                node0_addr = f'{socket.gethostbyname(socket.gethostname())}'
            
            node0_port = str(node0_port).lower()
            if node0_port == 'auto':
                sock = socket.socket()
                sock.bind(('', 0))
                _, node0_port = sock.getsockname()
                sock.close()
            
            node0_addr_port = f'tcp://{node0_addr}:{node0_port}'
            with open(temp_f_path, 'w') as fp:
                json.dump(node0_addr_port, fp)
            print(Fore.LIGHTBLUE_EX + f'{time_str()}[rk00] node0_addr_port: {node0_addr_port} (saved at \'{temp_f_path}\')')
        else:
            time.sleep(3 + rank * 0.1)
            while not os.path.exists(temp_f_path):
                print(Fore.CYAN + f'{time_str()}[rk{rank:02d}] try to read node0_addr_port')
                time.sleep(0.5)
            with open(temp_f_path, 'r') as fp:
                node0_addr_port = json.load(fp)
            print(Fore.LIGHTBLUE_EX + f'{time_str()}[rk{rank:02d}] node0_addr_port obtained')
        
        self.backend, self.node0_addr_port, self.world_size, self.rank = backend, node0_addr_port, world_size, rank
        self.node0_addr = self.node0_addr_port[self.node0_addr_port.find('//') + 2:self.node0_addr_port.rfind(':')]
        self.ngpus_per_node, self.dev_idx = torch.cuda.device_count(), ...
    
    # def initialize(self):
        dist.init_process_group(
            backend=self.backend, init_method=self.node0_addr_port,
            world_size=self.world_size, rank=self.rank
        )
        self.dev_idx: int = int(os.environ['SLURM_LOCALID'])            # equals to rank % gres_gpu
        torch.cuda.set_device(self.dev_idx)
        
        print(Fore.CYAN + f'{time_str()}[dist init] rank[{self.rank:02d}]: node0_addr_port={self.node0_addr_port}, gres_gpu(ngpus_per_node)={self.ngpus_per_node}, dev_idx={self.dev_idx}')
        
        assert torch.distributed.is_initialized()
        
        self.barrier()
        if self.is_master():
            os.remove(self.temp_f_path)
            print(Fore.LIGHTBLUE_EX + f'{time_str()}[rk00] removed temp file: \'{self.temp_f_path}\'')
    
    def finalize(self) -> None:
        print(f'{Fore.CYAN + time_str()}[dist finalize] rank[{self.rank:02d}]')

    def is_master(self):
        return self.rank == 0
    
    def get_world_group(self):
        return TorchDistManager.WORLD_GROUP
    
    def new_group(self, ranks: List[int]):
        return dist.new_group(ranks=ranks)
    
    def barrier(self) -> None:
        dist.barrier()
    
    def allreduce(self, t: torch.Tensor, group_idx_or_handler=None) -> None:
        if group_idx_or_handler is None:
            group_idx_or_handler = TorchDistManager.WORLD_GROUP
        if not t.is_cuda:
            cu = t.detach().cuda()
            dist.all_reduce(cu, group=group_idx_or_handler)
            t.copy_(cu.cpu())
        else:
            dist.all_reduce(t, group=group_idx_or_handler)
    
    def broadcast(self, t: torch.Tensor, rank_in_the_group: int, group_idx_or_handler=None) -> None:
        if group_idx_or_handler is None:
            group_idx_or_handler = TorchDistManager.WORLD_GROUP
        if not t.is_cuda:
            cu = t.detach().cuda()
            dist.broadcast(cu, src=rank_in_the_group, group=group_idx_or_handler)
            t.copy_(cu.cpu())
        else:
            dist.broadcast(t, src=rank_in_the_group, group=group_idx_or_handler)
    
    def dist_fmt_vals(self, val, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
        ts = torch.zeros(self.world_size)
        ts[self.rank] = val
        self.allreduce(ts)
        if fmt is None:
            return ts
        return [fmt % v for v in ts.numpy().tolist()]