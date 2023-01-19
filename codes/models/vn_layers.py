import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# EPS = 1e-6 # Original (VNN) EPS value
EPS = 1e-8 # Updated EPS value by GraphONet
EPS2 = 1e-12 # Secondary EPS value added by GraphONet. For use in VNLinearLeakyReLU and HNLinearLeakyReLU.

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
                    mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out



class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2, bn=False, apply_leaky_relu=True):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.apply_leaky_relu = apply_leaky_relu

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        if bn:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        self.bn=bn

        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)


    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # BatchNorm
        if self.bn:
            p = self.batchnorm(p)
        if self.apply_leaky_relu:
            # LeakyReLU
            d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
            dotprod = (p * d).sum(2, keepdims=True)

            # Original VNN code:
            mask = (dotprod >= 0).float()
            d_norm_sq = (d*d).sum(2, keepdims=True)
            x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS2))*d))

            # # GraphONet modifications in below comments. Should not have any effect in practice.
            # # - An inverted "mask" variable is used, and the expression for calculating x_out is simplified, bringing out "p" from all weighted / masked terms.
            # # - The d_norm_sq calculation is calculated differently, but appears to do the same thing.
            # mask = (dotprod < 0).float()
            # d_norm_sq = torch.pow(torch.norm(d, 2, dim=2, keepdim=True),2)
            # x_out = p - (mask) * (1-self.negative_slope) * (dotprod / (d_norm_sq + EPS2)) * d
        else:
            x_out = p
        return x_out


class HNLinearLeakyReLU(nn.Module):
    def __init__(self, v_in_channels, v_out_channels, s_in_channels=0, s_out_channels=0, dim=5, share_nonlinearity=False, negative_slope=0.2, bn=False, bias=True, scale_equivariance=False, s2v_norm_averaged_wrt_channels=True, s2v_norm_p=1, apply_leaky_relu=True):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.apply_leaky_relu = apply_leaky_relu

        self.is_s2v = self.s_in_channels > 0 and self.v_out_channels > 0

        self.map_to_feat = nn.Linear(v_in_channels, v_out_channels, bias=False)
        if bn:
            self.batchnorm = VNBatchNorm(v_out_channels, dim=dim)
        self.bn=bn
        if self.apply_leaky_relu:
            if share_nonlinearity == True:
                self.map_to_dir = nn.Linear(v_in_channels, 1, bias=False)
            else:
                self.map_to_dir = nn.Linear(v_in_channels, v_out_channels, bias=False)

        if s_in_channels > 0:
            # scalar
            if s_out_channels > 0:
                self.ss = nn.Linear(s_in_channels, s_out_channels, bias=bias)
            if self.is_s2v:
                self.sv = nn.Linear(s_in_channels, v_out_channels, bias=bias)
            if bn: # todo xuyaogai
                self.s_bn = nn.BatchNorm1d(s_out_channels)
        if s_out_channels > 0:
            self.v2s = VNStdFeature(v_in_channels, dim=dim, ver=1, reduce_dim2=True, regularize=True, scale_equivariance=scale_equivariance)
            self.vs = nn.Linear(v_in_channels, s_out_channels, bias=bias)

        self.s_in_channels = s_in_channels
        self.s_out_channels = s_out_channels
        self.v_in_channels = v_in_channels
        self.v_out_channels = v_out_channels

        if self.is_s2v:
            self.s2v_norm_averaged_wrt_channels = s2v_norm_averaged_wrt_channels
            self.s2v_norm_p = s2v_norm_p

        self.scale_equivariance = scale_equivariance

        assert self.v_in_channels > 0, 'A non-trivial group-action on the input, in particular v_in_channels > 0, is required for any non-trivial (non-invariant) equivariant layer. While we could implement a rotation-invariant scalar->scalar layer, this might as well be done by an ordinary (point-wise) linear layer.'


    def forward(self, x, s=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        s: scalar point features of shape [B, N_feat, N_samples, ...]
        '''

        # Linear
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)

        if s is not None:
            if self.is_s2v:
                # NOTE: sv maps scalar to scalars.
                # - This vector of scalars is then normalized with its absolute sum or absolute mean (depending on s2v_norm_averaged_wrt_channels). (absolute values assuming s2v_norm_p=1)
                # - Next, these scalars are multiplicated with corresponding output vector features p, effectively scaling them.
                sv = self.sv(s.transpose(1, -1)).transpose(1, -1).unsqueeze(2)
                if self.s2v_norm_averaged_wrt_channels:
                    p = p * sv / (sv.norm(p=self.s2v_norm_p, dim=1, keepdim=True) / self.v_out_channels + EPS)
                else:
                    p = p * sv / (sv.norm(p=self.s2v_norm_p, dim=1, keepdim=True)                       + EPS)
            #p = p * F.sigmoid(sv)

            if self.s_out_channels > 0:
                ss = self.ss(s.transpose(1, -1)).transpose(1, -1)
                vs = self.vs(self.v2s(x)[0].transpose(1, -1)).transpose(1, -1)
                s_out = ss + vs
                if self.apply_leaky_relu:
                    s_out = F.leaky_relu(s_out, self.negative_slope)
                if self.bn:
                    s_out = self.s_bn(s_out)
            else:
                s_out = None
        else:
            if self.s_out_channels > 0:
                vs = self.vs(self.v2s(x)[0].transpose(1, -1)).transpose(1, -1)
                s_out = vs
                if self.apply_leaky_relu:
                    s_out = F.leaky_relu(s_out, self.negative_slope)
            else:
                s_out = None

        # BatchNorm
        if self.bn:
            p = self.batchnorm(p)

        if self.apply_leaky_relu:
            # LeakyReLU
            d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
            d_norm_sq = torch.pow(torch.norm(d, 2, dim=2, keepdim=True), 2)
            dotprod = (p * d).sum(2, keepdims=True)
            mask = (dotprod < 0).float()
            # d_norm_sq = (d * d).sum(2, keepdims=True)
            # x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            #             mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS2)) * d))
            x_out = p - (mask) * (1 - self.negative_slope) * (dotprod / (d_norm_sq + EPS2)) * d
        else:
            x_out = p
        return x_out, s_out


class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, bn_mode='norm',
                 negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.bn_mode = bn_mode
        self.negative_slope = negative_slope

        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity,
                                      negative_slope=negative_slope)

        # BatchNorm
        self.bn_mode = bn_mode
        if bn_mode != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=bn_mode)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.bn_mode != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim, random_sign_flip=False):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        self.random_sign_flip = random_sign_flip
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        # norm = torch.norm(x,p=2,dim=2,keepdim=False) # NOTE: This is the norm calculation in the GraphONet implementation. They appear to have removed the EPS addition to the norm, perhaps by accident.
        norm = torch.norm(x, dim=2) + EPS

        if self.random_sign_flip:
            # Unlike the original VNN, a random sign flip is applied on the vector norms before applying the conventional BN layer.
            dims = norm.shape
            mask = torch.randint(0, 2, dims, device=norm.device).float()*2-1
            norm = norm * mask
            #norm[..., :norm.shape[-1]//2] = - norm[..., :norm.shape[-1]//2]

        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)

        if self.random_sign_flip:
            # After the BN layer, take the absolute value after all:
            norm = torch.abs(norm)
            norm_bn = torch.abs(norm_bn)

        x = x / norm * norm_bn

        return x




class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        # index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        # x_max = x[index_tuple]
        x_max = torch.gather(x, -1, idx.expand(x.shape[:-1]).unsqueeze(-1)).squeeze(-1)
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


# NOTE: Invarization layer.
class VNStdFeature(nn.Module): # also for HNStdFeature
    '''
    ver==0: old
    ver==1: z dir :== mean at N_feat
    ver==2: length of
    '''
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, ver=0, scale_equivariance=False, reduce_dim2=False, regularize=True): # todo regularize changed 1102
        super().__init__()
        self.ver = ver
        self.dim = dim
        if self.ver==0:
            self.normalize_frame = normalize_frame

            self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, share_nonlinearity=share_nonlinearity,
                                         negative_slope=negative_slope)
            self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim, share_nonlinearity=share_nonlinearity,
                                         negative_slope=negative_slope)
            if normalize_frame:
                self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
            else:
                self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)
        self.scale_equivariance=scale_equivariance
        self.reduce_dim2=reduce_dim2 # only work with ver1
        self.regularize = regularize
        if self.scale_equivariance: self.regularize = True
        self.in_channels = in_channels
    def forward(self, x, s=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        if self.ver == 0:
            z0 = x
            z0 = self.vn1(z0)
            z0 = self.vn2(z0)
            z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

            if self.normalize_frame:
                # make z0 orthogonal. u2 = v2 - proj_u1(v2)
                v1 = z0[:, 0, :]
                # u1 = F.normalize(v1, dim=1)
                v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
                u1 = v1 / (v1_norm + EPS)
                v2 = z0[:, 1, :]
                v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
                # u2 = F.normalize(u2, dim=1)
                v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
                u2 = v2 / (v2_norm + EPS)

                # compute the cross product of the two output vectors
                u3 = torch.cross(u1, u2)
                z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
            else:
                z0 = z0.transpose(1, 2)
        elif self.ver==1:
            z0 = torch.mean(x, dim=1, keepdim=True)
            if not self.reduce_dim2: # always
                shape = x[:, :3, ...].shape
                z0 = z0.expand(shape)
            z0 = z0.transpose(1, 2)
        # elif self.ver==2: # Not good
        #     z0 = None
        #     x_std = torch.sum(torch.pow(x, exponent=2), dim=2, keepdim=True).expand(x.shape).contiguous()


        if self.ver==0 or self.ver==1:
            if self.dim == 4:
                x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
            elif self.dim == 3:
                x_std = torch.einsum('bij,bjk->bik', x, z0)
            elif self.dim == 5:
                x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        if self.regularize:
            x_std = x_std / (z0.transpose(1,2).norm(p=2, dim=2, keepdim=True) + EPS)

            if self.scale_equivariance:
                # scale_norm = torch.norm(x_std, p=2, dim=2, keepdim=True).mean(dim=1, keepdim=True) # old no need to take norm if
                #scale_norm = torch.norm(x_std, p=1, dim=1, keepdim=True) / self.in_channels#.mean(dim=1, keepdim=True)
                scale_norm = torch.mean(torch.abs(x_std), dim=1, keepdim=True)
                x_std = x_std / (scale_norm + EPS)
                # if s is not None:
                #     s = s / (scale_norm.squeeze(2) + EPS )
        if self.reduce_dim2:
            assert self.ver==1
            x_std = x_std.squeeze(2)
        if s is not None:
            assert self.reduce_dim2
            x_std = torch.cat((x_std, s), dim=1)
        return x_std, z0
    # ver1 reduce dim2 B,nfeat,1,xxx => B,nfeat,xxx
    # ver1 not reduce dim2 B,nfeat,3,xxx
    # ver0 B,nfeat,3,xxx
