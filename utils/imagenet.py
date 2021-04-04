import pathlib
import warnings

try:
    import mc
except ImportError:
    pass

import io
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def pil_loader(img_bytes):
    # warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    buff = io.BytesIO(img_bytes)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


class ImageNetDataset(Dataset):
    # /mnt/lustre/share/images
    def __init__(self, root: str, train, transform, download=False, read_from='mc'):
        root = pathlib.Path(root)
        tr_va_root, tr_va_meta = root / 'train', root / 'meta' / 'train.txt'
        te_root, te_meta = root / 'val', root / 'meta' / 'val.txt'
        
        tu = (tr_va_root, tr_va_meta) if train else (te_root, te_meta)
        root_dir, meta_file = str(tu[0]), str(tu[1])
        self.root_dir = root_dir
        self.transform = transform
        self.read_from = read_from
        
        with open(meta_file) as f:
            lines = f.readlines()
        
        self.num_data = len(lines)
        self.metas = []
        for line in lines:
            img_path, label = line.rstrip().split()
            img_path = os.path.join(self.root_dir, img_path)
            self.metas.append((img_path, int(label)))
        self.metas = tuple(self.metas)
        self.targets = tuple(m[1] for m in self.metas)
        
        self.read_from = read_from
        self.initialized = False
    
    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = '/mnt/lustre/share/memcached_client/server_list.conf'
            client_config_file = '/mnt/lustre/share/memcached_client/client.conf'
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True
    
    def _init_ceph(self):
        if not self.initialized:
            # self.s3_client = ceph.S3Client()
            self.initialized = True
    
    def _init_petrel(self):
        if not self.initialized:
            # self.client = Client(enable_mc=True)
            self.initialized = True
    
    def read_file(self, filepath):
        if self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filepath, value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        elif self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(filepath, dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'ceph':
            self._init_ceph()
            value = self.s3_client.Get(filepath)
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'petrel':
            self._init_petrel()
            value = self.client.Get(filepath)
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(filepath, dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))
        
        return filebytes
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        img_path, label = self.metas[idx]
        img = pil_loader(self.read_file(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetDataset120(ImageNetDataset):
    def __init__(self, root, train, transform, download=False, read_from='mc'):
        super(ImageNetDataset120, self).__init__(root, train, transform, download, read_from)
        idx120 = [452, 664, 329, 559, 320, 167, 570, 241, 780, 695, 993, 706, 778, 710, 473, 323, 20, 217, 960, 541, 623, 676, 850, 487, 876, 577, 155, 140, 331, 884, 866, 400, 514, 159, 245, 758, 797, 586, 739, 491, 833, 913, 966, 419, 901, 872, 469, 666, 547, 790, 44, 987, 552, 997, 163, 156, 195, 436, 421, 725, 93, 110, 141, 281, 549, 708, 214, 975, 615, 830, 102, 716, 657, 744, 779, 131, 295, 859, 490, 69, 237, 444, 562, 663, 326, 413, 636, 511, 488, 47, 166, 72, 940, 571, 973, 860, 911, 304, 82, 711, 818, 434, 521, 381, 429, 734, 187, 81, 852, 655, 875, 573, 772, 904, 524, 299, 607, 856, 449, 603]
        assert len(set(idx120)) == 120
        
        # train size: 153487, test size: 6000
        me = list(filter(lambda tu: tu[1] in idx120, self.metas))
        self.metas = []
        for img_path, label in me:
            label = idx120.index(label)
            self.metas.append((img_path, label))
        
        self.metas = tuple(self.metas)
        self.targets = tuple(m[1] for m in self.metas)
        self.num_data = len(self.targets)
        assert self.num_data == (153487 if train else 6000)

# def create_imagenet_dataloaders(cfg_dataset, num_workers, batch_size, input_size=224, test_resize=256):
#     """
#     build training dataloader for ImageNet
#     """
#     cfg_train = cfg_dataset[data_type]
#     # build dataset
#     # NVIDIA dali preprocessing
#     dataset = ImageNetDataset(
#         root_dir=cfg_train['root_dir'],
#         meta_file=cfg_train['meta_file'],
#         read_from='mc',
#         transform=None
#     )
#
#     # build sampler
#     cfg_train['sampler']['kwargs'] = {}
#     cfg_dataset['dataset'] = dataset
#     sampler = build_sampler(cfg_train['sampler'], cfg_dataset)
#
#     # build dataloader
#     # NVIDIA dali pipeline
#     tr_kw = {'colorjitter': [0.2, 0.2, 0.2, 0.1]}
#     va_te_kw = {'size': test_resize}
#     tr_va_root, tr_va_meta = '/mnt/lustre/share/images/train/', '/mnt/lustre/share/images/meta/train.txt'
#     te_root, te_meta = '/mnt/lustre/share/images/val/', '/mnt/lustre/share/images/meta/val.txt'
#
#     kw = tr_kw
#     pp_cls = ImageNetTrainPipeV2 or ImageNetValPipeV2
#     root, meta = tr_va_root, tr_va_meta
#     pipeline = pp_cls(
#         data_root=root,
#         data_list=meta,
#         sampler=sampler,
#         crop=input_size,
#         **kw
#     )
#     loader = DaliDataloader(
#         pipeline=pipeline,
#         batch_size=batch_size,
#         epoch_size=len(sampler),
#         num_threads=num_workers,
#     )
#
#     return loader
