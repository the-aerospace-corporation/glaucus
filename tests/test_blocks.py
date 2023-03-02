'''ensure blocks are working'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest
import torch
from hypothesis import settings, given, strategies as st

from glaucus import GlaucusNet, blockgen, FullyConnected, GBlock


class TestParams(unittest.TestCase):
    '''autoencoders should operate over all valid params'''
    @settings(deadline=None, max_examples=100)
    @given(
        exponent=st.integers(min_value=2, max_value=14),
        steps=st.integers(min_value=1, max_value=8),
        filters_mid=st.integers(min_value=1, max_value=100)
    )
    def test_io_glaucusnet(self, exponent, steps, filters_mid):
        '''design works on a variety of spatial sizes'''
        spatial_dim = 2**exponent
        # for spatial_dim in 2**np.arange(8, 14):
        encoder_blocks = blockgen(steps=steps, spatial_in=spatial_dim, spatial_out=8, filters_in=2, filters_out=filters_mid, mode='encoder')
        decoder_blocks = blockgen(steps=steps, spatial_in=8, spatial_out=spatial_dim, filters_in=filters_mid, filters_out=2, mode='decoder')
        encoder = GlaucusNet(mode='encoder', blocks=encoder_blocks, spatial_dim=spatial_dim)
        decoder = GlaucusNet(mode='decoder', blocks=decoder_blocks, spatial_dim=spatial_dim)
        trash_x = torch.randn(3, 2, spatial_dim)
        trash_y = decoder(encoder(trash_x))
        self.assertEqual(trash_x.shape, trash_y.shape)

    @given(
        spatial_exponent=st.integers(min_value=6, max_value=14),
        filters_in=st.integers(min_value=1, max_value=128),
        filters_out=st.integers(min_value=1, max_value=128),
        stride=st.integers(min_value=1, max_value=8),
        expand_ratio=st.integers(min_value=1, max_value=10),
        squeeze_ratio=st.integers(min_value=1, max_value=10),
        kernel_size=st.integers(min_value=1, max_value=31),
    )
    def test_io_gblock(self, spatial_exponent, filters_in, filters_out, stride, expand_ratio, squeeze_ratio, kernel_size):
        spatial_size = 2**spatial_exponent
        if kernel_size % 2 == 0:
            # kernel_size must be odd
            kernel_size += 1
        if kernel_size < stride:
            # blockgen would not allow this
            return
        while squeeze_ratio > filters_in * expand_ratio:
            # blockgen would not allow this
            squeeze_ratio -= 1
        squeeze_ratio = max(filters_in * expand_ratio, squeeze_ratio)
        blk = GBlock(
            filters_in=filters_in, filters_out=filters_out,
            stride=stride, kernel_size=kernel_size,
            expand_ratio=expand_ratio, squeeze_ratio=squeeze_ratio
        )
        trash_x = torch.randn(2, filters_in, spatial_size)
        trash_y = blk(trash_x)
        spatial_out = int((spatial_size - kernel_size + 2 * blk.pad_size) / stride + 1)
        # check for output shape correct
        self.assertEqual(trash_y.shape, (2, filters_out, spatial_out))
        loss = torch.nn.MSELoss()(trash_y, torch.randn(trash_y.shape))
        # will raise RuntimeError here if there is an issue with backprop
        loss.backward()

    @given(
        exponent_in=st.integers(min_value=2, max_value=14),
        exponent_out=st.integers(min_value=2, max_value=14),
        steps=st.integers(min_value=1, max_value=5),
        quantize_in=st.booleans(), quantize_out=st.booleans(),
        use_dropout=st.booleans()
    )
    def test_io_fc(self, exponent_in, exponent_out, steps, quantize_in, quantize_out, use_dropout):
        '''block should work with a variety of configs'''
        size_in, size_out = exponent_in**2, exponent_out**2
        autoencoder = FullyConnected(
            size_in=size_in, size_out=size_out,
            steps=steps, quantize_in=quantize_in, quantize_out=quantize_out
        )
        trash_x = torch.randn(3, size_in)
        trash_y = autoencoder(trash_x)
        self.assertEqual(trash_y.shape[1], size_out)
