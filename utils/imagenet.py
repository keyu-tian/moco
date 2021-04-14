import pathlib
import warnings
from collections import defaultdict

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
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
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
    
    def get_untransformed_image(self, idx):
        return pil_loader(self.read_file(self.metas[idx][0]))
    
    def __len__(self):
        return self.num_data
    
    def __getitem__(self, idx):
        img_path, label = self.metas[idx]
        img = pil_loader(self.read_file(img_path))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


_idx_1300images = [
    457, 180, 796, 81, 83, 239, 423, 552, 84, 632, 960, 720, 633, 41, 93, 364, 207, 330, 555, 350, 13, 655, 123, 603, 170, 909, 777, 781, 762, 360, 904, 888, 8, 429, 882, 878, 498, 879,
    291, 151, 910, 367, 874, 578, 666, 37, 371, 902, 434, 713, 386, 428, 862, 840, 366, 196, 44, 86, 593, 716, 479, 951, 257, 787, 102, 599, 919, 59, 514, 486, 515, 283, 533, 246, 915,
    121, 60, 770, 851, 452, 380, 173, 219, 967, 208, 406, 407, 404, 629, 355, 117, 228, 730, 256, 575, 72, 799, 735, 737, 820, 692, 883, 66, 446, 956, 726, 313, 241, 274, 805, 293, 732,
    887, 788, 830, 522, 357, 111, 504, 792, 859, 505, 760, 473, 965, 512, 250, 715, 979, 379, 38, 761, 69, 463, 548, 818, 900, 929, 634, 222, 100, 227, 640, 362, 837, 402, 614, 661,
    971, 279, 581, 340, 369, 963, 758, 654, 162, 284, 306, 273, 990, 412, 943, 92, 642, 937, 269, 21, 977, 258, 435, 893, 759, 46, 78, 79, 517, 400, 329, 195, 748, 403, 710, 220, 941,
    876, 295, 764, 22, 565, 440, 769, 374, 42, 976, 450, 561, 543, 668, 973, 866, 318, 5, 184, 847, 519, 833, 325, 119, 711, 928, 547, 310, 563, 681, 405, 242, 617, 864, 191, 54, 146,
    554, 899, 939, 573, 451, 783, 608, 627, 528, 868, 616, 294, 998, 267, 780, 855, 225, 229, 105, 508, 619, 89, 624, 703, 414, 786, 553, 437, 12, 458, 36, 296, 280, 215, 867, 682, 287,
    537, 659, 691, 132, 694, 482, 253, 106, 557, 974, 513, 957, 506, 996, 63, 203, 171, 757, 570, 959, 745, 932, 82, 276, 602, 334, 986, 884, 934, 625, 118, 154, 895, 187, 916, 991,
    535, 922, 683, 489, 592, 725, 804, 500, 373, 292, 684, 734, 756, 393, 309, 470, 566, 779, 368, 378, 445, 48, 172, 775, 695, 299, 425, 277, 823, 924, 993, 323, 749, 248, 209, 319,
    39, 50, 849, 324, 142, 112, 767, 670, 410, 27, 572, 817, 628, 278, 10, 978, 282, 813, 953, 18, 697, 333, 539, 927, 272, 488, 57, 532, 107, 388, 518, 776, 315, 845, 345, 944, 988,
    449, 647, 308, 95, 157, 591, 597, 336, 199, 127, 128, 255, 97, 316, 87, 153, 903, 265, 936, 702, 448, 94, 305, 870, 381, 707, 143, 982, 595, 911, 997, 650, 281, 637, 980, 124, 64,
    952, 968, 524, 765, 589, 763, 962, 178, 77, 70, 160, 846, 2, 270, 658, 806, 286, 298, 189, 827, 768, 115, 302, 873, 433, 842, 836, 26, 413, 605, 383, 108, 226, 803, 460, 101, 660,
    865, 540, 858, 213, 356, 326, 408, 200, 739, 459, 600, 665, 65, 705, 907, 556, 751, 652, 755, 395, 415, 920, 314, 133, 155, 861, 832, 558, 376, 214, 636, 641, 493, 73, 538, 249,
    391, 680, 349, 961, 657, 363, 881, 218, 419, 618, 464, 177, 793, 839, 384, 432, 948, 444, 588, 156, 427, 232, 875, 61, 317, 606, 23, 985, 260, 417, 312, 235, 852, 342, 372, 247,
    411, 424, 32, 223, 443, 844, 230, 935, 148, 674, 604, 733, 972, 421, 159, 917, 621, 455, 351, 496, 0, 801, 816, 116, 718, 322, 17, 651, 808, 56, 736, 687, 698, 549, 523, 109, 321,
    835, 125, 685, 140, 834, 74, 584, 743, 422, 301, 819, 871, 721, 815, 169, 598, 231, 33, 938, 441, 186, 237, 476, 461, 139, 671, 690, 385, 750, 648, 886, 966, 216, 898, 332, 931,
    626, 15, 47, 149, 611, 897, 161, 709, 544, 212, 918, 431, 999, 338, 717, 136, 797, 774, 545, 699, 90, 954, 530, 244, 6, 693, 234, 71, 958, 290, 970, 942, 908, 35, 484, 571, 646,
    843, 477, 587, 114, 638, 701, 120, 719, 49, 52, 430, 11, 752, 91, 185, 174, 394, 773, 359, 377, 828, 245, 669, 987, 930, 263, 192, 525, 420, 494, 438, 341, 396, 933, 880, 814, 831,
    825, 14, 311, 387, 7, 4, 30, 643, 297, 179, 204, 520, 994, 122, 848, 738, 672, 612, 487, 397, 574, 300, 401, 288, 889, 673, 16, 398, 912, 25, 442, 850, 182, 88, 480, 76, 807, 529,
    995, 656, 339, 471, 343, 785, 511, 331, 468, 320, 31, 562, 348, 856, 29, 328, 905, 361, 104, 58, 475, 307, 354, 141, 622, 462, 337, 795, 576, 347, 40, 542, 947, 700, 766, 509, 559,
    289, 964, 447, 259, 923, 176, 266, 285, 389, 344, 983, 992, 67, 3, 620, 365, 20, 134, 251, 579, 913, 677, 510, 586, 352, 201, 271, 516, 534, 138, 688, 456, 984, 975, 704, 467, 210,
    150, 145, 416, 478, 469, 236, 949, 254, 233, 45, 527, 564, 454, 131, 945, 645, 28, 492, 890, 261, 55, 453, 113, 163, 75, 85, 607, 644, 126, 483, 877, 594, 243, 679, 560, 744, 541,
    474, 824, 667, 466, 746, 68, 485, 198, 24, 894, 495, 609, 791, 754, 129, 778, 197, 144, 664, 436, 382, 490, 205, 211, 1, 353, 615, 193, 19, 794, 950, 224, 472, 809, 358, 110, 327,
    568, 53, 217, 99, 896, 639, 303, 9, 802, 784, 696, 546, 829, 569, 304, 202, 264, 346, 96, 981, 526, 822, 989, 863, 238, 582, 137, 130, 853, 580, 375, 741, 34, 497, 135, 800, 502,
    613, 742, 601, 649, 240, 80, 399, 275, 370, 790, 955
]  # 895 classes
_target_num_per_cls = 1024


class SubImageNetDataset(ImageNetDataset):
    def __init__(self, num_classes: int, root, train, transform, download=False, read_from='mc'):
        super(SubImageNetDataset, self).__init__(root, train, transform, download, read_from)
        assert num_classes <= len(_idx_1300images), f'{num_classes} > {len(_idx_1300images)}(len(_idx_1300images))'
        
        selected_labels = _idx_1300images[:num_classes]
        # original: 1281167
        count = defaultdict(int)
        me = list(filter(lambda tu: tu[1] in selected_labels, self.metas))
        self.metas = []
        for img_path, label in me:
            label = selected_labels.index(label)
            if train and count[label] == _target_num_per_cls:
                continue
            count[label] += 1
            self.metas.append((img_path, label))
        
        self.metas = tuple(self.metas)
        self.targets = tuple(m[1] for m in self.metas)
        self.num_data = len(self.targets)
        assert self.num_data == num_classes * (_target_num_per_cls if train else 50)

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
