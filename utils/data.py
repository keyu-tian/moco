import numpy as np
from torch.utils.data.sampler import Sampler

from utils.misc import ints_ceil


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else ints_ceil(dataset_len, batch_size)
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            indices = np.concatenate((indices, tails))
            if self.shuffle:
                np.random.shuffle(indices)
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep
