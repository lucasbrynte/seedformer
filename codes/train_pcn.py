'''
==============================================================

SeedFormer: Point Cloud Completion
-> Training on PCN dataset

==============================================================

Author: Haoran Zhou
Date: 2022-5-31

==============================================================
'''


import argparse
import os
import random
import numpy as np
import torch
import json
import time
from torch.utils.tensorboard import SummaryWriter
import utils.datasets
from utils.helpers import set_seed
from utils.helpers import get_dflt_conf
from easydict import EasyDict as edict
from importlib import import_module
from pprint import pprint
from manager import Manager


TRAIN_NAME = os.path.splitext(os.path.basename(__file__))[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           Arguments 
#       \******************/
#

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--arch_model', type=str, default='seedformer_dim128', help='Model to use.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=False, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
parser.add_argument('--lr', type=float, dest='lr', help='', default=None)
parser.add_argument('--batch-size', type=int, dest='batch_size', help='', default=None)
parser.add_argument('--attn-channel', type=str, dest='attn_channel', help="'1', '2', 'both', or 'none'", default='2')
parser.add_argument('--pos-features-feat-extractor', type=str, dest='pos_features_feat_extractor', help="'abs', 'rel', or 'none'", default='abs')
parser.add_argument('--pos-features-up-layers', type=str, dest='pos_features_up_layers', help="'abs', 'rel', 'rel_nofeatmax', 'none', or 'none_deeper'", default='abs')
args = parser.parse_args()


def PCNConfig():

    #######################
    # Configuration for PCN
    #######################

    __C                                              = get_dflt_conf()
    cfg                                              = __C

    #
    # Dataset
    #
    __C.DATASET                                      = edict()
    # Dataset Options: Completion3D, ShapeNet (=PCN), ShapeNet55, ShapeNetCars, Completion3DPCCT
    __C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
    __C.DATASET.VAL_DATASET                          = 'ShapeNet'
    __C.DATASET.TEST_DATASET                         = 'ShapeNet'
    __C.DATASET.VALIDATE_ON_TEST                     = False

    #
    # Network
    #
    __C.NETWORK.UPSAMPLE_FACTORS                     = [1, 4, 8] # 512 * (1 * 4 * 8) = 16384 pts

    #
    # Train
    #
    # __C.TRAIN.LR_DECAY                               = 100 # NOTE: Seedformer paper states 100 rather than 150, and 100 is used when training on ShapeNet-55.
    __C.TRAIN.LR_DECAY                               = 150

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    train_data_loader, val_data_loaders = utils.datasets.init_train_val_dataloaders(cfg)

    # Set up folders for logs and checkpoints
    timestr = time.strftime('_Log_%Y_%m_%d_%H_%M_%S', time.localtime())
    cfg.DIR.OUT_PATH = os.path.join(cfg.DIR.OUT_PATH, TRAIN_NAME+timestr)
    cfg.DIR.CHECKPOINTS = os.path.join(cfg.DIR.OUT_PATH, 'checkpoints')
    cfg.DIR.LOGS = cfg.DIR.OUT_PATH
    cfg.DIR.TB = os.path.join(cfg.DIR.OUT_PATH, 'tb')
    print('Saving outdir: {}'.format(cfg.DIR.OUT_PATH))
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    # save config file
    pprint(cfg)
    config_filename = os.path.join(cfg.DIR.LOGS, 'config.json')
    with open(config_filename, 'w') as file:
        json.dump(cfg, file, indent=4, sort_keys=True)

    # Save Arguments
    torch.save(args, os.path.join(cfg.DIR.LOGS, 'args_training.pth'))

    # Create tensorboard writers
    os.makedirs(cfg.DIR.TB, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(cfg.DIR.TB))

    #######################
    # Prepare Network Model
    #######################

    Model = import_module(args.net_model)
    model = Model.__dict__[args.arch_model](
        attn_channel = cfg.NETWORK.ATTN_CHANNEL,
        pos_features_feat_extractor = cfg.NETWORK.POS_FEATURES_FEAT_EXTRACTOR,
        pos_features_up_layers = cfg.NETWORK.POS_FEATURES_UP_LAYERS,
        up_factors = cfg.NETWORK.UPSAMPLE_FACTORS,
    )
    # print(model)
    if cfg.PARALLEL.MULTIGPU:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.DistributedDataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # load existing model
    if 'WEIGHTS' in cfg.CONST:
        print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        print('Recover complete. Current epoch = #%d; best metrics = %s.' % (checkpoint['epoch_index'], checkpoint['best_metrics']))


    ##################
    # Training Manager
    ##################

    manager = Manager(model, cfg)

    # Start training
    manager.train(model, train_data_loader, val_data_loaders, cfg, tb_writer)


def test_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    ########################
    # Load Train/Val Dataset
    ########################

    test_data_loader = utils.datasets.init_test_dataloader(cfg)

    # Path for pretrained model
    if args.pretrained == '':
        list_trains = os.listdir(cfg.DIR.OUT_PATH)
        list_pretrained = [train_name for train_name in list_trains if train_name.startswith(TRAIN_NAME+'_Log')]
        if len(list_pretrained) != 1:
            raise ValueError('Find {:d} models. Please specify a path for testing.'.format(len(list_pretrained)))

        cfg.DIR.PRETRAIN = list_pretrained[0]
    else:
        cfg.DIR.PRETRAIN = args.pretrained


    # Set up folders for logs and checkpoints
    cfg.DIR.TEST_PATH = os.path.join(cfg.DIR.TEST_PATH, cfg.DIR.PRETRAIN)
    cfg.DIR.RESULTS = os.path.join(cfg.DIR.TEST_PATH, 'outputs')
    cfg.DIR.LOGS = cfg.DIR.TEST_PATH
    print('Saving outdir: {}'.format(cfg.DIR.TEST_PATH))
    if not os.path.exists(cfg.DIR.RESULTS):
        os.makedirs(cfg.DIR.RESULTS)


    #######################
    # Prepare Network Model
    #######################

    Model = import_module(args.net_model)
    model = Model.__dict__[args.arch_model](
        attn_channel = cfg.NETWORK.ATTN_CHANNEL,
        pos_features_feat_extractor = cfg.NETWORK.POS_FEATURES_FEAT_EXTRACTOR,
        pos_features_up_layers = cfg.NETWORK.POS_FEATURES_UP_LAYERS,
        up_factors = cfg.NETWORK.UPSAMPLE_FACTORS,
    )
    if cfg.PARALLEL.MULTIGPU:
        model = torch.nn.DataParallel(model)
        # model = torch.nn.DistributedDataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # load pretrained model
    cfg.CONST.WEIGHTS = os.path.join(cfg.DIR.OUT_PATH, cfg.DIR.PRETRAIN, 'checkpoints', 'ckpt-best.pth')
    print('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model'])

    ##################
    # Training Manager
    ##################

    manager = Manager(model, cfg)

    # Start training
    manager.test(cfg, model, test_data_loader, outdir=cfg.DIR.RESULTS if args.output else None)
        

if __name__ == '__main__':
    # Check python version
    #seed = 1
    #set_seed(seed)
    
    print('cuda available ', torch.cuda.is_available())

    # Init config
    cfg = PCNConfig()

    if args.lr is not None:
        cfg.TRAIN.LEARNING_RATE = args.lr
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    assert args.attn_channel in ['1', '2', 'both', 'none']
    cfg.NETWORK.ATTN_CHANNEL = args.attn_channel
    assert args.pos_features_feat_extractor in ['abs', 'rel', 'none']
    cfg.NETWORK.POS_FEATURES_FEAT_EXTRACTOR = args.pos_features_feat_extractor
    assert args.pos_features_up_layers in ['abs', 'rel', 'rel_nofeatmax', 'none', 'none_deeper']
    cfg.NETWORK.POS_FEATURES_UP_LAYERS = args.pos_features_up_layers

    if not args.test and not args.inference:
        train_net(cfg)
    else:
        if args.test:
            test_net(cfg)
        else:
            inference_net(cfg)

