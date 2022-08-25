'''assure blocks are working'''
import unittest
import numpy as np
import torch

from glaucus import GlaucusNet, blockgen, FullyConnected


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

    def test_spatial_io_fc(self):
        '''forward works on a variety of spatial sizes'''
        for size_in in 2**np.arange(8, 14):
            for size_out in 2**np.arange(6, 12):
                autoencoder = FullyConnected(size_in=size_in, size_out=size_out)
                trash_x = torch.randn(32, size_in)
                trash_y = autoencoder(trash_x)
                self.assertEqual(trash_y.shape[1], size_out)