#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''assure autoencoders are working as expected'''
import unittest
import numpy as np
import torch

from glaucus import GlaucusNet, blockgen


class TestParams(unittest.TestCase):
    '''autoencoders should operate over all valid params'''
    def test_spatial_io_glaucus(self):
        '''forward works on a variety of spatial sizes'''
        for spatial_dim in 2**np.arange(8, 14):
            encoder_blocks = blockgen(steps=6, spatial_in=spatial_dim, spatial_out=8, filters_in=2, filters_out=64, mode='encoder', verbose=0)
            decoder_blocks = blockgen(steps=6, spatial_in=8, spatial_out=spatial_dim, filters_in=64, filters_out=2, mode='decoder', verbose=0)
            encoder = GlaucusNet(mode='encoder', blocks=encoder_blocks, spatial_dim=spatial_dim)
            decoder = GlaucusNet(mode='decoder', blocks=decoder_blocks, spatial_dim=spatial_dim)
            trash_x = torch.randn(32, 2, spatial_dim)
            trash_y = decoder(encoder(trash_x))
            self.assertEqual(trash_x.shape, trash_y.shape)
