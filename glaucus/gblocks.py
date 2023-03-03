# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging
import math
from collections import namedtuple
import numpy as np

import torch
import lightning as pl
from torch import nn

from .layers import DropConnect

log = logging.getLogger(__name__)

# BlockArgs are used to specify a Glaucus architecture, which is thankfully generated smartly by `blockgen`.
BlockArgs = namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'filters_in', 'filters_out', 'expand_ratio', 'squeeze_ratio'])


def blockgen(
        spatial_in:int=4096, spatial_out:int=8,
        filters_in:int=2, filters_out:int=64,
        expand_ratio:int=4, squeeze_ratio:int=4,
        steps:int=6, mode:str='encoder'):
    '''
    smartly generate block sequence

    * Geometrically increase/decrease filters and increase repeats.
    * Distribute spatial scaling across units
    * Geometrically increase kernel size from 1.5% to 15% per [1][2][3][4].

    After a week long experiment in Oct 2021 I determined that you always want
    strides to be larger earlier in either encoder or decoder network.
    Similarly for number of repeat steps you want more repeats as the network gets deeper.

    References
    ----------
    [1] https://arxiv.org/abs/2104.00298
    [2] https://arxiv.org/abs/1905.11946
    [3] https://arxiv.org/abs/2109.07161
    [4] https://arxiv.org/abs/1905.02244
    '''
    # functions that round UP to either even or odd
    round_to_odd = lambda val: np.ceil(val) // 2 * 2 + 1
    round_to_even = lambda val: np.ceil(val / 2) * 2
    # calculate required strides; smartly distribute scaling
    strides = np.ones(steps, dtype=int)
    doubles = np.abs(np.log2(spatial_in/spatial_out))
    assert doubles % 1 == 0, 'spatial values must be 2**n'
    for dbx in range(int(doubles)):
        # smartly distribute strides to achieve spatial size change
        strides[dbx % steps] *= 2
    # calculate the fraction of spatial time-series covered by kernel
    # between geomspace and linspace per [1][2][3][4]
    kernel_percent = (np.linspace(.015, .15, steps) + np.geomspace(.015, .15, steps))/2
    if mode == 'decoder':
        kernel_percent = np.flip(kernel_percent)
    # calculate the filters per step
    filter_steps = np.geomspace(filters_in, filters_out, steps+1).round().astype(int)

    # calculate the repeats per block; 15 max, for now use 8
    repeat_steps = np.geomspace(2, 8, steps).round().astype(int)
    blocks = []
    params = np.zeros(steps) # only used for logging
    for sdx in range(steps):
        if mode == 'encoder':
            spatial_in /= strides[sdx]
        else:
            spatial_in *= strides[sdx]
        kernel_size = int(max(3, round_to_odd(kernel_percent[sdx]*spatial_in)))
        while kernel_size < strides[sdx]:
            # this is bad
            # https://ezyang.github.io/convolution-visualizer/index.html
            # https://distill.pub/2016/deconv-checkerboard/
            log.warning('bumping kernel_size to prevent checkerboarding resulting from kernel_size < stride')
            kernel_size += 2

        block = BlockArgs(
            num_repeat=repeat_steps[sdx],
            kernel_size=kernel_size,
            stride=strides[sdx],
            filters_in=filter_steps[sdx],
            filters_out=filter_steps[sdx+1],
            expand_ratio=expand_ratio,
            squeeze_ratio=squeeze_ratio,
        )
        log.info(f'{block}')
        params[sdx] += block.filters_in * block.filters_out # conv_reshape
        params[sdx] += block.filters_in * block.filters_in * block.expand_ratio * block.kernel_size # conv_expand
        params[sdx] += (block.filters_in * block.expand_ratio)**2 # conv_middle
        params[sdx] += (block.filters_in * block.expand_ratio)**2 / block.squeeze_ratio # linear_squeeze x2
        params[sdx] += block.filters_in * block.expand_ratio * block.filters_out # conv_tail
        log.debug(f'params={params[sdx]:.0f}, out_shape=({block.filters_out:.0f}, {spatial_in:.0f})')
        blocks += [block]
    return blocks


# defaults
ENCODER_BLOCKS = blockgen(steps=6, spatial_in=4096, spatial_out=8, filters_in=2, filters_out=64, mode='encoder')
DECODER_BLOCKS = blockgen(steps=6, spatial_in=8, spatial_out=4096, filters_in=64, filters_out=2, mode='decoder')


class GBlock(pl.LightningModule):
    '''
    Inspired by ResNetBlock, MBConvBlock, & FusedMBConvBlock

    Parameters
    ----------
    filters_in : int
        Channel dim of NCL input.
    filters_out : int
        Channel dim of NCL output.
    mode : str, default 'encoder'
        Either 'encoder' or 'decoder', modifies where scaling occurs and
        determines whether 'stride' will decimate or interpolate.
    stride : int, default 1
        When stride is > 1, will decimate or interpolate the output length depending on 'mode'.
    expand_ratio : int, default 4
        Expansion ratio for input channel count.
    squeeze_ratio : float, default 4
        Squeeze excitation ratio applied to expanded channel count.
    kernel_size : int, default 7
        Kernel size should be adjusted based on network depth.
        Note even kernel sizes not supported due to padding difficulties.
    drop_connect_rate : float, default 0.2
        Proability of dropped connections in residual layer.
        With a deep network design this value should be zero early and scale up to max value.

    Key Features
    ------------
    * Scalable Width & Depth [6][7]
    * Fused Inverted Residual Blocks [10][6][7]
    * Stochastic Depth [5]
    * Squeeze & Excitation [8] on expanded channels
    * DropConnect instead of Dropout [9]

    References
    ----------
    [1] https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    [2] https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
    [3] https://www.fatalerrors.org/a/efficientnet-v2-papers-and-code-analysis.html
    [4] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    [5] https://arxiv.org/abs/1603.09382
    [6] https://arxiv.org/abs/2104.00298
    [7] https://arxiv.org/abs/1905.11946
    [8] https://arxiv.org/abs/1709.01507
    [9] http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf
    [10] https://arxiv.org/abs/1905.02244

    Design
    ------
    filters_in ➡ filters_ex ➡ filters_sq ➡ filters_out
    only apply stride on first repeat of each block

    original_stack = conv ➡ bn ➡ activ ➡ conv ➡ bn ➡ +x ➡ activ
    preactiv_stack = bn ➡ activ ➡ conv ➡ bn ➡ activ ➡ conv ➡ +x
    fused_stack = conv_ex ➡ bn ➡ activ ➡ +se ➡ conv ➡ bn ➡ +x

    In EfNetV2, depth is scaled from (1, 5) and width is scaled (1, 4)

    Major Changes
    -------------
    * GBlock design will always add residual unlike efnet.
    * Will include Expand-Squeeze from EfNet.
    * Will include Fused blocks from EFNetV2.
    * Squeeze size calculated on expanded filters, not input filters

    *DANGER* even kernel sizes not quite supported; padding nightmares
    '''
    def __init__(self,
                 filters_in:int, filters_out:int, mode:str='encoder',
                 stride:int=1, drop_connect_rate:float=0.2,
                 expand_ratio:int=4, squeeze_ratio:int=4, kernel_size:int=7):
        super().__init__()
        assert mode in ['encoder', 'decoder']
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.expand_ratio = expand_ratio
        self.squeeze_ratio = squeeze_ratio
        self.mode = mode
        self.stride = stride
        self.bn_mom = 1e-2 # better than torch default
        self.bn_eps = 1e-3 # better than torch default
        self.filters_ex = self.filters_in * self.expand_ratio
        self.filters_sq = int(max(1, round(self.filters_ex / self.squeeze_ratio)))
        self.is_reshaped = self.filters_in != self.filters_out or self.stride != 1
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            # odd kernel size
            self.pad_size = (kernel_size - 1) // 2
        else:
            # this is not quite working
            self.pad_size = (kernel_size // 2) - 1
        # special arguments for convolution steps
        expand_args = {}
        reshape_args = {}
        # self._zpad = torch.nn.ZeroPad2d(padding=(1,0)) # using for 1D when even kernel
        # Define all layers
        self._activ = nn.SiLU(inplace=True) # Activation De-Jour

        # adjust for encoder/decoder
        if self.mode == 'encoder':
            # more depth, less samples
            Conv1d = nn.Conv1d
        else:
            # less depth, more samples
            Conv1d = nn.ConvTranspose1d

        expand_args['kernel_size'] = self.kernel_size
        expand_args['stride'] = self.stride
        expand_args['padding'] = self.pad_size
        if self.mode == 'decoder':
            # tricky deconvolution padding required
            expand_args['output_padding'] = self.stride - 1

        if self.is_reshaped:
            # we need this to adapt the identity to the correct shape
            # using as few weights as possible. Large experiment on 2021-12-27
            # determined k=1 was better than attempting to use k=stride
            if self.mode == 'decoder' and self.stride != 1:
                # tricky deconvolution padding
                reshape_args['output_padding'] = self.stride - 1
            self._reshape = nn.Sequential(
                Conv1d(self.filters_in, self.filters_out, stride=self.stride, bias=False, kernel_size=1, **reshape_args),
                nn.BatchNorm1d(self.filters_out),
            )

        # Expansion Layers
        self._conv_expand = Conv1d(self.filters_in, self.filters_ex, bias=False, **expand_args)
        self._bn0 = nn.BatchNorm1d(self.filters_ex, momentum=self.bn_mom, eps=self.bn_eps)

        if self.squeeze_ratio != 1:
            # Squeeze-Excitation Layers
            self._avgpool = nn.AdaptiveAvgPool1d(1)
            self._se_reduce = nn.Linear(self.filters_ex, self.filters_sq)
            self._se_expand = nn.Linear(self.filters_sq, self.filters_ex)

        self._conv_tail = Conv1d(self.filters_ex, self.filters_out, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm1d(self.filters_out, momentum=self.bn_mom, eps=self.bn_eps)
        self._dropconnect = DropConnect(drop_connect_rate)

        # Block Info
        log.debug('GBlock{}(stride={}, filters=({},{},{},{}))'.format(
                'Enc' if self.mode == 'encoder' else 'Dec', self.stride,
                self.filters_in, self.filters_ex, self.filters_sq, self.filters_out,
                ))

    def forward(self, x):
        identity = x
        if self.is_reshaped:
            # we might have different output shape
            identity = self._reshape(identity)
        # Fused Expansion Phase (Inverted Bottleneck)
        x = self._conv_expand(x)
        x = self._bn0(x)
        x = self._activ(x)
        # Squeeze-Excitation Phase
        if self.squeeze_ratio != 1:
            x_squeezed = self._avgpool(x).squeeze()
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._activ(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed).unsqueeze(-1)
            # doing this in-place causes backprop err
            x = x * torch.sigmoid(x_squeezed)
        # Pointwise Convolution Phase
        x = self._conv_tail(x)
        x = self._bn1(x)
        # drop connection on some decreasing fraction of x only while training
        x = self._dropconnect(x)
        # Skip Connection
        x += identity
        return x


class GlaucusNet(nn.Module):
    def __init__(self,
                 blocks=ENCODER_BLOCKS, mode:str='encoder',
                 width_coef:float=1.0, depth_coef:float=1.0,
                 spatial_dim:int=4096, drop_connect_rate:float=0.2,
        ) -> None:
        super().__init__()
        assert isinstance(blocks, list), 'should be list of BlockArgs'
        assert len(blocks) > 0, 'cannot have zero blocks'
        assert mode in ['encoder', 'decoder']
        self.width_coef = width_coef
        self.depth_coef = depth_coef
        self.mode = mode
        # drop connect rate is scaled from 0 to drop_connect_rate through block network
        self.drop_connect_rate = drop_connect_rate
        if self.mode == 'decoder':
            # need to reshape flat input into (filters_in, spatial_dim) for blocks
            self.filters_in = blocks[0].filters_in
        # construct blocks
        self._blocks = nn.ModuleList()
        drop_num, drop_denom = 0, -1 # keep track to scale drop_connect_rate
        for bdx, block_args in enumerate(blocks):
            # count up how many total blocks there actually are with change to dept_coef
            drop_denom += self.round_repeats(block_args.num_repeat, self.depth_coef)
        for bdx, block_args in enumerate(blocks):
            # account for width & depth scaling
            num_repeat = self.round_repeats(block_args.num_repeat, self.depth_coef)
            if bdx == 0:
                # fixed stem size on first block input
                blk_filters_in = block_args.filters_in
            else:
                blk_filters_in = self.round_filters(block_args.filters_in, self.width_coef)
            if bdx == len(blocks)-1:
                # fixed stem size on last block output
                blk_filters_out = block_args.filters_out
            else:
                blk_filters_out = self.round_filters(block_args.filters_out, self.width_coef)
            self._blocks.append(GBlock(
                filters_in=blk_filters_in,
                filters_out=blk_filters_out,
                expand_ratio=block_args.expand_ratio,
                squeeze_ratio=block_args.squeeze_ratio,
                kernel_size=block_args.kernel_size,
                stride=block_args.stride, # only first block handles stride
                drop_connect_rate=self.drop_connect_rate*(drop_num/drop_denom),
                mode=self.mode,
            ))
            drop_num += 1
            # handle repeats
            for _ in range(num_repeat - 1):
                self._blocks.append(GBlock(
                    filters_in=blk_filters_out,
                    filters_out=blk_filters_out,
                    expand_ratio=block_args.expand_ratio,
                    squeeze_ratio=block_args.squeeze_ratio,
                    kernel_size=block_args.kernel_size,
                    stride=1, # no scale change in repeats
                    drop_connect_rate=self.drop_connect_rate*(drop_num/drop_denom),
                    mode=self.mode,
                ))
                drop_num += 1

    @staticmethod
    def round_filters(base, multiplier):
        '''vary width based on multiplier'''
        return int(round(base * multiplier))

    @staticmethod
    def round_repeats(base, multiplier):
        '''vary depth based on multiplier'''
        return int(math.ceil(base * multiplier))

    def extract_features(self, x):
        for block in self._blocks:
            x = block(x)
        return x

    def forward(self, x):
        if self.mode == 'decoder':
            # un-flatten to feed into decoder
            x = x.view(x.size(0), self.filters_in, -1)
        x = self.extract_features(x)
        if self.mode == 'encoder':
            # pooling & final linear layer
            # x = self._avgpool(x).squeeze(-1)
            # TODO: Note to self, if reimplementing avgpool, use self.dropout_rate
            x = x.flatten(start_dim=1)
        return x
