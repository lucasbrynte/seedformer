
import json
import logging
import numpy as np
import random
import torch.utils.data.dataset
import open3d as o3d
import utils.data_transforms
from enum import Enum, unique
from tqdm import tqdm
from utils.io import IO
import os

label_mapping = {
    3: '03001627',
    6: '04379243',
    5: '04256520',
    1: '02933112',
    4: '03636649',
    2: '02958343',
    0: '02691156',
    7: '04530566'
}


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


code_mapping = {
    'plane': '02691156',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'lamp': '03636649',
    'couch': '04256520',
    'table': '04379243',
    'watercraft': '04530566',
}


def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return ptcloud


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1

        # random select one sample per shape for training
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] -
                                      1) if self.options['shuffle'] else 0

        # load required data
        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                assert ri == 'partial_cloud'
                # A list of renderings is always accepted, even if n_renderings == 1
                assert len(file_path) == self.options['n_renderings']
                file_path = file_path[rand_idx]
            elif ri == 'partial_cloud':
                # The only case for which we don't expect a list
                assert self.options['n_renderings'] == 1
            # print(file_path)
            data[ri] = IO.get(file_path).astype(np.float32)

        # apply transforms
        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDatasetWrapper(object):
    """
    PCN dataset: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset),
                                        n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset(
            {
                'n_renderings': n_renderings,
                'required_items': ['partial_cloud', 'gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):

                # Previously, 'test' has been considered a special case with a slightly different directory structure. The directory structure of the test data does however now appear identical to train / val.
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    # NOTE: All partial renderings are added to a single sample, but when accessing a dataset element, only one rendering will be (randomly) sampled:
                    'partial_cloud_path': [
                        cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH %
                        (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path':
                    cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


class ShapeNetCarsDatasetWrapper(ShapeNetDatasetWrapper):
    """
    ShapeNet only on car category
    """
    def __init__(self, cfg):
        super(ShapeNetCarsDatasetWrapper, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] == '02958343'
        ]


class Completion3DDatasetWrapper(object):
    """
    Completion3D: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = [
            'partial_cloud'
        ] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset(
            {
                'required_items': required_items,
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback':
                'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        elif subset == DatasetSubset.VAL:
            return utils.data_transforms.Compose([{
                'callback':
                'ScalePoints',
                'parameters': {
                    'scale': 0.85
                },
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback':
                'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'ToTensor',
                'objects': ['partial_cloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH %
                    (subset, dc['taxonomy_id'], s),
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


class Completion3DPCCTDatasetWrapper(Completion3DDatasetWrapper):
    """
    Dataset Completion3D containing only plane, car, chair, table
    """
    def __init__(self, cfg):
        super(Completion3DPCCTDatasetWrapper, self).__init__(cfg)

        # Remove other categories except couch, chairs, car, lamps
        cat_set = {'02691156', '03001627', '02958343',
                   '04379243'}  # plane, chair, car, table
        # cat_set = {'04256520', '03001627', '02958343', '03636649'}
        self.dataset_categories = [
            dc for dc in self.dataset_categories
            if dc['taxonomy_id'] in cat_set
        ]


class KittiDatasetWrapper(object):
    """
    KITTI: extracted car objects
    """
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({
            'required_items': required_items,
            'shuffle': False
        }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback':
            'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback':
            'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        # Collect file list
        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' %
                         (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path':
                    cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info(
            'Complete collecting files of the dataset. Total files: %d' %
            len(file_list))
        return file_list


# ShapeNet-55/34
# Ref: https://github.com/yuxumin/PoinTr/blob/master/datasets/ShapeNet55Dataset.py
class ShapeNet55Dataset(torch.utils.data.dataset.Dataset):
    """
    ShapeNet55 dataset: return complete clouds, partial clouds are generated online
    """
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms
        self.cache = dict()

    def __len__(self):
        return len(self.file_list)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)
            # shapenet55
            data[ri] = self.pc_norm(data[ri])
            data[ri] = torch.from_numpy(data[ri]).float()

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNet55DatasetWrapper(object):
    """
    ShapeNet55: get dataset file list
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = None
        return ShapeNet55Dataset(
            {
                'required_items': ['gtcloud'],
                'shuffle': subset == DatasetSubset.TRAIN
            }, file_list, transforms)

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""

        # Load the dataset indexing file
        with open(
                os.path.join(cfg.DATASETS.SHAPENET55.CATEGORY_FILE_PATH,
                             subset + '.txt'), 'r') as f:
            lines = f.readlines()

        # Collect file list
        file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            file_list.append({
                'taxonomy_id':
                taxonomy_id,
                'model_id':
                model_id,
                'gtcloud_path':
                cfg.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH % (line),
            })

        print('Complete collecting files of the dataset. Total files: %d' %
              len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Wrapper Mapping = //////////////////////////////////////////// #

DATASET_WRAPPER_MAPPING = {
    'Completion3D': Completion3DDatasetWrapper,
    'Completion3DPCCT': Completion3DPCCTDatasetWrapper,
    'ShapeNet': ShapeNetDatasetWrapper,
    'ShapeNetCars': ShapeNetCarsDatasetWrapper,
    'KITTI': KittiDatasetWrapper,
    'ShapeNet55': ShapeNet55DatasetWrapper,
}  # yapf: disable


# ////////////////////////////////////// = Dataloader Initialization Functions = ////////////////////////////////////// #

def init_test_dataloader(cfg):
    test_dataset_wrapper = DATASET_WRAPPER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset_wrapper.get_dataset(DatasetSubset.TEST),
        batch_size = 1,
        num_workers = cfg.PARALLEL.NUM_WORKERS,
        collate_fn = collate_fn,
        pin_memory = True,
        shuffle = False,
        drop_last = False,
    )
    return test_data_loader


def init_train_val_dataloaders(cfg):
    train_dataset_wrapper = DATASET_WRAPPER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset_wrapper.get_dataset(DatasetSubset.TRAIN),
        batch_size = cfg.TRAIN.BATCH_SIZE,
        num_workers = cfg.PARALLEL.NUM_WORKERS,
        collate_fn = collate_fn,
        pin_memory = True,
        shuffle = True,
        drop_last = False,
    )

    val_data_loaders = {}
    if cfg.DATASET.VAL_DATASET is not None:
        val_dataset_wrapper = DATASET_WRAPPER_MAPPING[cfg.DATASET.VAL_DATASET](cfg)
        val_data_loaders['VAL'] = torch.utils.data.DataLoader(
            dataset = val_dataset_wrapper.get_dataset(DatasetSubset.VAL),
            batch_size = cfg.TRAIN.BATCH_SIZE,
            num_workers = cfg.PARALLEL.NUM_WORKERS//2,
            collate_fn = collate_fn,
            pin_memory = True,
            shuffle = False,
            drop_last = False,
        )
    if cfg.DATASET.VALIDATE_ON_TEST:
        val_data_loaders['TEST'] = init_test_dataloader(cfg)

    return train_data_loader, val_data_loaders
