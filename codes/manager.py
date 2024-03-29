# --------------------------------------------------------
# SeedFormer
# Copyright (c) 2022 Intelligent Media Pervasive, Recognition & Understanding Lab of NJU - All Rights Reserved
# Licensed under The MIT License [see LICENSE for details]
# Written by Haoran Zhou
# --------------------------------------------------------

'''
==============================================================

SeedFormer: Point Cloud Completion
-> Training/Testing Manager

==============================================================

Author: Haoran Zhou
Date: 2022-5-31

==============================================================
'''


import os
import numpy as np
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
import time

import utils.helpers
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import get_loss
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer
#       \***************/
#

class Manager:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, cfg):
        """
        Initialize parameters and start training/testing
        :param model: network object
        :param cfg: configuration object
        """

        ############
        # Parameters
        ############
        
        # training dataset
        self.dataset = cfg.DATASET.TRAIN_DATASET

        # Epoch index
        self.epoch = 0

        # Create the optimizers
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=cfg.TRAIN.LEARNING_RATE,
                                           weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                           betas=cfg.TRAIN.BETAS)

        # lr scheduler
        # NOTE: In general, before PyTorch 1.13, the following issue requires the first scheduler to be the last one initialized:
        # https://github.com/pytorch/pytorch/issues/72874
        # PR merged for PT 1.13:
        # https://github.com/pytorch/pytorch/pull/72856
        # For LinearLR & StepLR, order does not seem to matter, at least not in the current scheme.
        self.scheduler_steplr = StepLR(self.optimizer, step_size=1, gamma=0.1 ** (1 / cfg.TRAIN.LR_DECAY))

        # OLD:
        # # NOTE: multiplier=1 implies special case of ramping up from 0 to base_lr.
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_EPOCHS,
        #                                       after_scheduler=self.scheduler_steplr)

        # NEW: (Now also avoid initial manual self.lr_scheduler.step() call)
        cfg.TRAIN.WARMUP_EPOCHS -= 1 # There is no longer any redundant epoch due to the double initial LR scheduler step (once by pytorch, once by user).
        self.scheduler_warmup = LinearLR(
            self.optimizer,
            start_factor = 1.0 / (cfg.TRAIN.WARMUP_EPOCHS+1),
            # NOTE! Important to not use any other end_factor than 1.0, without making sure we obtain expected behavior.
            # SequentialLR doesn't seem to guarantee "LR stitching", but instead, each scheduler resets at the same "base_lr".
            end_factor = 1.0,
            total_iters = cfg.TRAIN.WARMUP_EPOCHS,
        )
        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers = [self.scheduler_warmup, self.scheduler_steplr],
            milestones = [cfg.TRAIN.WARMUP_EPOCHS],
        )

        # record file
        self.train_record_file = open(os.path.join(cfg.DIR.LOGS, 'training.txt'), 'w')
        self.test_record_file = open(os.path.join(cfg.DIR.LOGS, 'testing.txt'), 'w')

        # eval metric
        self.best_metrics = float('inf')
        self.best_epoch = 0


    # Record functions
    def train_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.train_record_file:
            self.train_record_file.write(info + '\n')
            self.train_record_file.flush()

    def test_record(self, info, show_info=True):
        if show_info:
            print(info)
        if self.test_record_file:
            self.test_record_file.write(info + '\n')
            self.test_record_file.flush()

    def unpack_data(self, data):

        if self.dataset == 'ShapeNet':
            partial = data['partial_cloud']
            gt = data['gtcloud']
        elif self.dataset == 'ShapeNet55':
            # generate partial data online
            gt = data['gtcloud']
            _, npoints, _ = gt.shape
            partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
        else:
            raise ValueError('No method implemented for this dataset: {:s}'.format(self.dataset))

        return partial, gt


    def train(self, model, train_data_loader, val_data_loaders, cfg, tb_writer):
        init_epoch = 0
        steps = 0

        # training record file
        print('Training Record:')
        self.train_record('n_itr, cd_pc, cd_p1, cd_p2, cd_p3, partial_matching, grad_tot_absmean, grad_tot_absmax')
        print('Testing Record:')
        self.test_record('#epoch cdc cd1 cd2 partial_matching | cd3 | #best_epoch best_metrics')

        # # OLD:
        # # this zero gradient update is needed to avoid a warning message, issue #8.
        # # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/8
        # # In all likelihood, it does not have any practical consequence.
        # self.optimizer.zero_grad()
        # self.optimizer.step()

        # Training Start
        for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

            self.epoch = epoch_idx

            # timer
            epoch_start_time = time.time()

            model.train()

            # # OLD:
            # # Update learning rate
            # # Intentionally done before the first optimizer step (and implied lr_scheduler.get_lr() call).
            # # The author of the LR scheduler probably didn't realize that the .step() method of any LR scheduler is always called once by the constructor's call to ._initial_step().
            # # Consequently, in practice .step() is indeed called BEFORE every epoch, although the initial call is not meant to be manually carried out.
            # # A simple replacement of self.last_epoch by (self.last_epoch + 1) in all or parts of the code would likely result in the expected behavior, even when refraining from calling .step() before any optimizer.step().
            # self.lr_scheduler.step()

            # total cds
            total_cd_pc = 0
            total_cd_p1 = 0
            total_cd_p2 = 0
            total_cd_p3 = 0
            total_partial = 0

            batch_end_time = time.time()
            n_batches = len(train_data_loader)
            learning_rate = self.optimizer.param_groups[0]['lr']
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                pcds_pred = model(partial)

                loss_total, losses, gts = get_loss(pcds_pred, partial, gt, sqrt=True)

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                cd_p3_item = losses[3].item() * 1e3
                total_cd_p3 += cd_p3_item
                partial_item = losses[4].item() * 1e3
                total_partial += partial_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx

                # TB batch training record
                tb_writer.add_scalar('Train/Loss/Batch/cd_pc', cd_pc_item, n_itr)
                tb_writer.add_scalar('Train/Loss/Batch/cd_p1', cd_p1_item, n_itr)
                tb_writer.add_scalar('Train/Loss/Batch/cd_p2', cd_p2_item, n_itr)
                tb_writer.add_scalar('Train/Loss/Batch/cd_p3', cd_p3_item, n_itr)
                tb_writer.add_scalar('Train/Loss/Batch/partial_matching', partial_item, n_itr)

                if cfg.TRAIN.LOG_GRADIENTS:
                    flattened_gradients = utils.helpers.get_groupwise_flattened_gradients(model.module if cfg.PARALLEL.MULTIGPU else model)
                    for key, flattened_grad in flattened_gradients.items():
                        tb_writer.add_scalar('Train/Grad/Batch/{}_ABSMEAN'.format(key), torch.mean(torch.abs(flattened_grad)).item(), n_itr)
                        tb_writer.add_scalar('Train/Grad/Batch/{}_MAX'.format(key), torch.max(flattened_grad).item(), n_itr)
                    grad_total_absmean = torch.mean(torch.abs(torch.nn.utils.parameters_to_vector(flattened_gradients.values()))).item()
                    grad_total_absmax = torch.max(torch.abs(torch.nn.utils.parameters_to_vector(flattened_gradients.values()))).item()
                    tb_writer.add_scalar('Train/Grad/Batch/TOTAL_ABSMEAN'.format(key), grad_total_absmean, n_itr)
                    tb_writer.add_scalar('Train/Grad/Batch/TOTAL_ABSMAX'.format(key), grad_total_absmax, n_itr)

                # training record
                message = '{:d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(n_itr, cd_pc_item, cd_p1_item, cd_p2_item, cd_p3_item, partial_item)
                if cfg.TRAIN.LOG_GRADIENTS:
                    message += ' {:.4f} {:.4f}'.format(grad_total_absmean, grad_total_absmax)
                self.train_record(message, show_info=False)

                # # For fast debugging: Iterate only 1 batch:
                # break

            # NEW:
            # Update learning rate
            # NOTE: At the scheduler switch, SequentialLR.step() calls scheduler.step(0), triggering an EPOCH_DEPRECATION_WARNING. Confusing but harmless.
            self.lr_scheduler.step()

            # avg cds
            avg_cdc = total_cd_pc / n_batches
            avg_cd1 = total_cd_p1 / n_batches
            avg_cd2 = total_cd_p2 / n_batches
            avg_cd3 = total_cd_p3 / n_batches
            avg_partial = total_partial / n_batches

            epoch_end_time = time.time()

            # TB epoch training record
            tb_writer.add_scalar('Train/Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
            tb_writer.add_scalar('Train/Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
            tb_writer.add_scalar('Train/Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
            tb_writer.add_scalar('Train/Loss/Epoch/cd_p3', avg_cd3, epoch_idx)
            tb_writer.add_scalar('Train/Loss/Epoch/partial_matching', avg_partial, epoch_idx)
            tb_writer.add_scalar('Misc/Epoch/learning_rate', learning_rate, epoch_idx)

            # Training record
            self.train_record(
                '[Epoch %d/%d] LearningRate = %f EpochTime = %.3f (s) Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS, learning_rate, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2, avg_cd3, avg_partial]]))

            # Validate the current model
            val_scores = {}
            for val_data_id, val_data_loader in val_data_loaders.items():
                val_scores[val_data_id] = self.validate(cfg, model=model, val_data_loader=val_data_loader, val_data_id=val_data_id, tb_writer=tb_writer)
            if cfg.DATASET.VAL_DATASET is not None:
                # Validation data exists.
                main_val_score = val_scores['VAL']
            elif cfg.DATASET.VALIDATE_ON_TEST:
                # Use test data instead.
                main_val_score = val_scores['TEST']
            elif len(val_scores) == 1:
                # There is only one validation set, so we can unambiguously determine which dataset to calculate the score on.
                main_val_score = val_scores[val_data_id]
            else:
                # In this case, there is no validation dataset at all, or there are > 1, in which case it is ambiguous from which to determine the main score.
                main_val_score = None
            self.train_record('Testing scores = {:.4f}'.format(main_val_score))

            # Save checkpoints
            if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or (main_val_score is not None and main_val_score < self.best_metrics):
                self.best_epoch = epoch_idx

                file_names = []
                if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0:
                    file_names.append('ckpt-epoch-%03d.pth' % epoch_idx)
                if (main_val_score is not None and main_val_score < self.best_metrics):
                    file_names.append('ckpt-best.pth')

                for file_name in file_names:
                    output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
                    torch.save({
                        'epoch_index': epoch_idx,
                        'val_scores': val_scores,
                        'model': model.state_dict()
                    }, output_path)

                    print('Saved checkpoint to %s ...' % output_path)
                if (main_val_score is not None and main_val_score < self.best_metrics):
                    self.best_metrics = main_val_score

            # main_val_score is an ordinary python float, but numpy works fine for NaN detection:
            assert np.isfinite(main_val_score), 'Encountered NaN validation loss at epoch {}'.format(epoch_idx)

        # training end
        tb_writer.close()
        self.train_record_file.close()
        self.test_record_file.close()


    def validate(self, cfg, model=None, val_data_loader=None, val_data_id=None, tb_writer=None):
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(val_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter('cd3')

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(val_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=True)

                # get metrics
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3

                _metrics = [cd3]
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])
                test_metrics.update(_metrics)

        # Add validation results to TensorBoard
        if tb_writer is not None:
            val_id_str = 'Val_{}'.format(val_data_id) if val_data_id is not None else 'Val'
            tb_writer.add_scalar('{}/Loss/Epoch/cd_pc'.format(val_id_str), test_losses.avg(0), self.epoch)
            tb_writer.add_scalar('{}/Loss/Epoch/cd_p1'.format(val_id_str), test_losses.avg(1), self.epoch)
            tb_writer.add_scalar('{}/Loss/Epoch/cd_p2'.format(val_id_str), test_losses.avg(2), self.epoch)
            tb_writer.add_scalar('{}/Loss/Epoch/cd_p3'.format(val_id_str), test_losses.avg(3), self.epoch)
            tb_writer.add_scalar('{}/Loss/Epoch/partial_matching'.format(val_id_str), test_losses.avg(4), self.epoch)
            for i, metric in enumerate(test_metrics.items):
                try:
                    tb_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), self.epoch)
                except ZeroDivisionError:
                    # TODO: Find the root cause of this issue
                    pass

        # Record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message, show_info=False)

        return test_losses.avg(3)

    def test(self, cfg, model, test_data_loader, outdir, mode=None):

        if self.dataset == 'ShapeNet':
            self.test_pcn(cfg, model, test_data_loader, outdir)
        elif self.dataset == 'ShapeNet55':
            self.test_shapenet55(cfg, model, test_data_loader, outdir, mode)
        else:
            raise ValueError('No testing method implemented for this dataset: {:s}'.format(self.dataset))

    def test_pcn(self, cfg, model=None, test_data_loader=None, outdir=None):
        """
        Testing Method for dataset PCN
        """

        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # unpack data
                partial, gt = self.unpack_data(data)

                # forward
                pcds_pred = model(partial.contiguous())
                loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=True)

                # get loss
                cdc = losses[0].item() * 1e3
                cd1 = losses[1].item() * 1e3
                cd2 = losses[2].item() * 1e3
                cd3 = losses[3].item() * 1e3
                partial_matching = losses[4].item() * 1e3
                test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                # get all metrics
                _metrics = Metrics.get(pcds_pred[-1], gt)
                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                # output to file
                if outdir:
                    if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                        os.makedirs(os.path.join(outdir, taxonomy_id))
                    if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                        os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                    # save pred, gt, partial pcds 
                    pred = pcds_pred[-1]
                    for mm, model_name in enumerate(model_id):
                        output_file = os.path.join(outdir, taxonomy_id, model_name)
                        write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                        # output img files
                        img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                        output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                        output_img = (output_img*255).astype('uint8')
                        im = Image.fromarray(output_img)
                        im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)

    def test_shapenet55(self, cfg, model=None, test_data_loader=None, outdir=None, mode=None):
        """
        Testing Method for dataset shapenet-55/34
        """

        from models.utils import fps_subsample
        
        # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
        torch.backends.cudnn.benchmark = True

        # Eval settings
        crop_ratio = {
            'easy': 1/4,
            'median' :1/2,
            'hard':3/4
        }
        choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                  torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]

        # Switch models to evaluation mode
        model.eval()

        n_samples = len(test_data_loader)
        test_losses = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
        test_metrics = AverageMeter(Metrics.names())
        mclass_metrics = AverageMeter(Metrics.names())
        category_metrics = dict()

        # Start testing
        print('Start evaluating (mode: {:s}) ...'.format(mode))
        for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            #model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                # generate partial data online
                gt = data['gtcloud']
                _, npoints, _ = gt.shape
                
                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)

                    pcds_pred = model(partial.contiguous())
                    loss_total, losses, _ = get_loss(pcds_pred, partial, gt, sqrt=False) # L2

                    # get loss
                    cdc = losses[0].item() * 1e3
                    cd1 = losses[1].item() * 1e3
                    cd2 = losses[2].item() * 1e3
                    cd3 = losses[3].item() * 1e3
                    partial_matching = losses[4].item() * 1e3
                    test_losses.update([cdc, cd1, cd2, cd3, partial_matching])

                    # get all metrics
                    _metrics = Metrics.get(pcds_pred[-1], gt)
                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)

                    # output to file
                    if outdir:
                        if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                            os.makedirs(os.path.join(outdir, taxonomy_id))
                        if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                            os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                        # save pred, gt, partial pcds 
                        pred = pcds_pred[-1]
                        for mm, model_name in enumerate(model_id):
                            output_file = os.path.join(outdir, taxonomy_id, model_name+'_{:02d}'.format(partial_id))
                            write_ply(output_file + '_pred.ply', pred[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                            # output img files
                            img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                            output_img = pc_util.point_cloud_three_views(pred[mm, :].detach().cpu().numpy(), diameter=7)
                            output_img = (output_img*255).astype('uint8')
                            im = Image.fromarray(output_img)
                            im.save(img_filename)


        # Record category results
        self.train_record('============================ TEST RESULTS ============================')
        self.train_record('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

        for taxonomy_id in category_metrics:
            message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0)) 
            message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
            mclass_metrics.update(category_metrics[taxonomy_id].avg())
            self.train_record(message)

        self.train_record('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(['%.4f' % value for value in test_metrics.avg()]))
        self.train_record('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))

        # record testing results
        message = '#{:d} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f} | #{:d} {:.4f}'.format(self.epoch, test_losses.avg(0), test_losses.avg(1), test_losses.avg(2), test_losses.avg(4), test_losses.avg(3), self.best_epoch, self.best_metrics)
        self.test_record(message)


        return test_losses.avg(3)
