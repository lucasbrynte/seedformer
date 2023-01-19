# --------------------------------------------------------
# Copyright. All Rights Reserved
# --------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import random
from easydict import EasyDict as edict
from models.utils import fps_subsample


def get_dflt_conf():
    __C                                              = edict()
    cfg                                              = __C


    ####################################################################
    # Datasets
    #
    __C.DATASETS                                     = edict()

    # __C.DATASETS.COMPLETION3D                        = edict()
    # __C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
    # __C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
    # __C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'

    __C.DATASETS.SHAPENET                            = edict()
    __C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
    __C.DATASETS.SHAPENET.N_RENDERINGS               = 8
    __C.DATASETS.SHAPENET.N_POINTS                   = 2048
    __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/data/PCN/%s/partial/%s/%s/%02d.pcd'
    __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/data/PCN/%s/complete/%s/%s.pcd'
    # __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '<*PATH-TO-YOUR-DATASET*>/PCN/%s/partial/%s/%s/%02d.pcd'
    # __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '<*PATH-TO-YOUR-DATASET*>/PCN/%s/complete/%s/%s.pcd'

    __C.DATASETS.SHAPENET55                          = edict()
    __C.DATASETS.SHAPENET55.CATEGORY_FILE_PATH       = './datasets/ShapeNet55-34/ShapeNet-55/'
    __C.DATASETS.SHAPENET55.N_POINTS                 = 2048
    __C.DATASETS.SHAPENET55.COMPLETE_POINTS_PATH     = '<*PATH-TO-YOUR-DATASET*>/ShapeNet55/shapenet_pc/%s'


    ####################################################################
    # Parallelism
    #
    __C.PARALLEL                                     = edict()
    __C.PARALLEL.NUM_WORKERS                         = 8
    __C.PARALLEL.MULTIGPU                            = True


    ####################################################################
    # Constants
    #
    __C.CONST                                        = edict()
    __C.CONST.N_INPUT_POINTS                         = 2048


    ####################################################################
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = '../results'
    # __C.DIR.TEST_PATH                                = '../test'
    __C.DIR.TEST_PATH                                = '../results/test'
    # __C.CONST.WEIGHTS                                = None # 'ckpt-best.pth'  # specify a path to run test and inference


    ####################################################################
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [1, 4, 8] # 16384
    __C.NETWORK.ATTN_CHANNEL                         = '2'
    __C.NETWORK.POS_FEATURES_FEAT_EXTRACTOR          = 'abs'
    __C.NETWORK.POS_FEATURES_UP_LAYERS               = 'abs'
    # __C.NETWORK.VNN                                  = None
    __C.NETWORK.VNN                                  = edict()
    __C.NETWORK.VNN.ENABLED                          = True
    # __C.NETWORK.VNN.ENABLED                          = False
    __C.NETWORK.VNN.GROUP                            = 'SO2'
    # __C.NETWORK.VNN.GROUP                            = 'SO3'
    __C.NETWORK.VNN.VECTOR_DIM                       = {'SO2': 2, 'SO3': 3}[__C.NETWORK.VNN.GROUP]
    __C.NETWORK.VNN.SCALAR_FEAT_FRAC                 = 0.5
    __C.NETWORK.VNN.HYBRID_FEATURE_LAYER_SETTINGS    = edict(negative_slope=0.0, bias=True, scale_equivariance=False, s2v_norm_averaged_wrt_channels=True, s2v_norm_p=1)


    ####################################################################
    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 48
    __C.TRAIN.N_EPOCHS                               = 400
    __C.TRAIN.SAVE_FREQ                              = 25
    __C.TRAIN.LEARNING_RATE                          = 0.001
    # Continuous decay parameter for StepLR scheduler. Meaning: After this #epochs, LR will have decayed to a factor 0.1.
    __C.TRAIN.LR_DECAY                               = 100
    __C.TRAIN.WARMUP_EPOCHS                          = 20
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0


    ####################################################################
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'

    return cfg


def set_seed(seed, seed_python=True, seed_numpy=True, seed_pytorch=True):
    if seed_python:
        random.seed(seed)
    if seed_numpy:
        np.random.seed(seed)
    if seed_pytorch:
        torch.manual_seed(seed)
    # # Legacy pytorch functions for separate GPU seeding:
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def get_param_groups(model):
    param_groups = {}
    param_groups['featext'] = list(model.feat_extractor.parameters())
    param_groups['seedgen'] = list(model.seed_generator.parameters())
    for j, curr_uplayer in enumerate(model.up_layers):
        param_groups['uplayer{}'.format(j)] = list(curr_uplayer.parameters())
    assert set(model.parameters()) == set().union(*param_groups.values())
    return param_groups


def get_groupwise_flattened_gradients(model):
    param_groups = get_param_groups(model)
    flattened_gradients = { key: torch.nn.utils.parameters_to_vector(p.grad for p in param_group) for key, param_group in param_groups.items() }
    return flattened_gradients


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(8, 8))

    x, z, y = ptcloud.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    ax.axis('scaled')
    ax.view_init(30, 45)

    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    return img


def seprate_point_cloud(xyz,
                        num_points,
                        crop,
                        fixed_points=None,
                        padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _, n, c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None

    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop, list):
            num_crop = random.randint(crop[0], crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1),
                                     p=2,
                                     dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1,
                            descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0,
                                        idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps_subsample(input_data, 2048))
            CROP.append(fps_subsample(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    return input_data.contiguous(), crop_data.contiguous()
