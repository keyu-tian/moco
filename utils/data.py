from typing import NamedTuple, Tuple, Dict

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils.imagenet import ImageNetDataset, SubImageNetDataset, _target_num_per_cls
from utils.misc import ints_ceil


class _DatasetMeta(NamedTuple):
    img_ch: int
    train_val_set_size: int
    test_set_size: int
    img_size: int
    num_classes: int
    clz: Dataset.__class__
    mean_std: Tuple[tuple, tuple]


dataset_metas: Dict[str, _DatasetMeta] = {
    'imagenet': _DatasetMeta(
        img_ch=3,
        train_val_set_size=1281168,
        test_set_size=50000,
        img_size=224,
        num_classes=1000,
        clz=ImageNetDataset,
        mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ),
    'subimagenet': _DatasetMeta(
        img_ch=3,
        train_val_set_size=_target_num_per_cls,
        test_set_size=50,
        img_size=224,
        num_classes=0,
        clz=SubImageNetDataset,
        mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ),
    'cifar10': _DatasetMeta(
        img_ch=3,
        train_val_set_size=50000,
        test_set_size=10000,
        img_size=32,
        num_classes=10,
        clz=torchvision.datasets.CIFAR10,
        mean_std=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ),
    'cifar100': _DatasetMeta(
        img_ch=3,
        train_val_set_size=50000,
        test_set_size=10000,
        img_size=32,
        num_classes=100,
        clz=torchvision.datasets.CIFAR100,
        mean_std=((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ),
}


class InputPairSet(Dataset):
    def __init__(self, origin_dataset):
        self.origin_dataset = origin_dataset
        self.has_attr_data = hasattr(origin_dataset, 'data') and hasattr(origin_dataset.data, '__getitem__')
        if not self.has_attr_data:
            assert hasattr(origin_dataset, 'get_untransformed_image')
        assert hasattr(origin_dataset, 'transform') and origin_dataset.transform is not None
    
    def __len__(self):
        return len(self.origin_dataset)
    
    def __getitem__(self, index):
        if self.has_attr_data:
            pil_img = self.origin_dataset.data[index]
        else:
            pil_img = self.origin_dataset.get_untransformed_image(index)
        return self.origin_dataset.transform(pil_img), self.origin_dataset.transform(pil_img)


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, shuffle=True, drop_last=False, fill_last=False, seed=None):
        assert not (drop_last and fill_last)
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else ints_ceil(dataset_len, batch_size)
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = 1
        self.seed = seed
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        if self.shuffle:
            if self.seed is None:
                indices = torch.randperm(self.dataset_len)
            else:
                g = torch.Generator()
                g.manual_seed(self.seed * 100 + self.epoch)
                indices = torch.randperm(self.dataset_len, generator=g)
        else:
            indices = torch.arange(self.dataset_len)
        
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            indices = torch.cat((indices, tails), dim=0)
            if self.shuffle:
                if self.seed is None:
                    indices = indices[torch.randperm(indices.shape[0])]
                else:
                    g = torch.Generator()
                    g.manual_seed(self.seed * 1000 + self.epoch)
                    indices = indices[torch.randperm(indices.shape[0], generator=g)]
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.numpy().tolist())
    
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


if __name__ == '__main__':
    sp = InfiniteBatchSampler(50000, 512, shuffle=True, drop_last=False, fill_last=True, seed=0)
    n = 0
    it = iter(sp)
    for i in range(len(sp)):
        idx = next(it)
        n += len(idx)
        if i == 0:
            print(idx[:5])
    print(len(sp), n)
