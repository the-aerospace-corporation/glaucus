'''assure AEs are working'''
import unittest
import numpy as np
import torch

from glaucus import GlaucusAE, FullyConnectedAE


class TestAE(unittest.TestCase):
    def test_ae_roundtrip(self):
        '''the  output size should always be the same as the input size'''
        for AE in [GlaucusAE, FullyConnectedAE]:
            for domain in ['time', 'freq']:
                spatial_size = 4096
                trash_x = torch.randn(32, 2, spatial_size)
                ae = AE(spatial_size=spatial_size, domain=domain)
                trash_y, _ = ae(trash_x)
                self.assertEqual(trash_x.shape, trash_y.shape)

    def test_ae_quantization(self):
        '''If quantization enabled, should use quint8 as latent output'''
        for AE in [FullyConnectedAE, GlaucusAE]:
            for is_quantized in [True, False]:
                target = torch.quint8 if is_quantized else torch.float32
                spatial_size = 4096
                trash_x = torch.randn(32, 2, spatial_size)
                ae = AE(spatial_size=spatial_size, bottleneck_quantize=is_quantized)
                if is_quantized:
                    # this will prepare the quant/dequant layers
                    torch.quantization.prepare(ae, inplace=True)
                    # this applies the quantization coefficients within the bottleneck
                    torch.quantization.convert(ae.cpu(), inplace=True)
                _, trash_latent = ae(trash_x)
                self.assertEqual(trash_latent.dtype, target)