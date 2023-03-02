'''ensure AEs are working'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest
import torch

from glaucus import GlaucusAE, FullyConnectedAE


class TestAE(unittest.TestCase):
    def test_ae_roundtrip(self):
        '''the  output size should always be the same as the input size'''
        for AE in [GlaucusAE, FullyConnectedAE]:
            for data_format in ['ncl', 'nl']:
                for domain in ['time', 'freq']:
                    # note if we use a diff spatial_size, will need to gen new encoder & decoder bocks
                    spatial_size = 4096
                    if data_format == 'ncl':
                        trash_x = torch.randn(7, 2, spatial_size)
                    else:
                        trash_x = torch.randn(7, spatial_size, dtype=torch.complex64)
                    ae = AE(domain=domain, data_format=data_format)
                    trash_y, _ = ae(trash_x)
                    self.assertEqual(trash_x.shape, trash_y.shape)

    def test_ae_quantization(self):
        '''If quantization enabled, should use quint8 as latent output'''
        for AE in [FullyConnectedAE, GlaucusAE]:
            for data_format in ['ncl', 'nl']:
                for is_quantized in [True, False]:
                    target = torch.quint8 if is_quantized else torch.float32
                    # note if we use a diff spatial_size, will need to gen new encoder & decoder bocks
                    spatial_size = 4096
                    if data_format == 'ncl':
                        trash_x = torch.randn(7, 2, spatial_size)
                    else:
                        trash_x = torch.randn(7, spatial_size, dtype=torch.complex64)
                    ae = AE(bottleneck_quantize=is_quantized, data_format=data_format)
                    if is_quantized:
                        # this will prepare the quant/dequant layers
                        torch.quantization.prepare(ae, inplace=True)
                        # this applies the quantization coefficients within the bottleneck
                        torch.quantization.convert(ae.cpu(), inplace=True)
                    _, trash_latent = ae(trash_x)
                    self.assertEqual(trash_latent.dtype, target)

    def test_ae_backprop(self):
        '''catch errors during backpropagation'''
        for data_format in ['ncl', 'nl']:
            for AE in [FullyConnectedAE, GlaucusAE]:
                for is_quantized in [True, False]:
                    # note if we use a diff spatial_size, will need to gen new encoder & decoder bocks
                    spatial_size = 4096
                    if data_format == 'ncl':
                        trash_x = torch.randn(7, 2, spatial_size)
                    else:
                        trash_x = torch.randn(7, spatial_size, dtype=torch.complex64)
                    ae = AE(bottleneck_quantize=is_quantized, data_format=data_format)
                    if is_quantized:
                        # this will prepare the quant/dequant layers
                        torch.quantization.prepare(ae, inplace=True)
                        # this applies the quantization coefficients within the bottleneck
                        torch.quantization.convert(ae.cpu(), inplace=True)
                    loss, _ = ae.step((trash_x, None), 0)
                    # will raise RuntimeError here if there is an issue with backprop
                    loss.backward()
