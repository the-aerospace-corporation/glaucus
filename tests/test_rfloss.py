'''ensure RFLoss is working'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest
import torch

from glaucus import RFLoss


class TestRFLoss(unittest.TestCase):
    def _gen_x(self, batch_size=32, spatial_size=4096) -> None:
        self.alpha = torch.randn(batch_size, 2, spatial_size)
        self.omega = torch.randn(batch_size, 2, spatial_size)

    def test_weights(self):
        '''
        make sure returns correct number of values
        zero out weights for keys one by one
        should always return individual loss and the total loss
        '''
        weight_keys = ['weight_env', 'weight_fft', 'weight_align', 'weight_spec', 'weight_xcor']
        self._gen_x()
        for test_key in weight_keys:
            kwargs = {test_key: 1}
            for zero_key in weight_keys:
                if zero_key != test_key:
                    kwargs[zero_key] = 0

            criterion = RFLoss(**kwargs)
            _, metrics = criterion(self.alpha, self.omega)
            self.assertEqual(2, len(metrics))

    def test_naive_case(self):
        '''loss between identical signals should be near zero'''
        for spatial_size in 2**torch.arange(8, 14):
            self._gen_x(spatial_size=spatial_size)
            criterion = RFLoss(spatial_size=spatial_size)
            total_loss, _ = criterion(self.alpha, self.alpha)
            self.assertAlmostEqual(total_loss.numpy(), 0, places=5)

    def test_spec_loss(self):
        '''
        the spec_loss weight is scaled by spatial_size,
        so make sure that's working as intended
        '''
        for spatial_size in 2**torch.arange(8, 14):
            self._gen_x(spatial_size=spatial_size)
            criterion = RFLoss(spatial_size=spatial_size)
            # should be around 1 for uncorrelated inputs
            _, metrics = criterion(self.alpha, self.omega)
            self.assertAlmostEqual(metrics['spec_loss'].numpy(), 0.858, places=1)
            # should be 0 for equal inputs
            _, metrics = criterion(self.alpha, self.alpha)
            self.assertAlmostEqual(metrics['spec_loss'].numpy(), 0)
            # should be 0 for inversely correlated inputs due to absolute value
            _, metrics = criterion(self.alpha, -self.alpha)
            self.assertAlmostEqual(metrics['spec_loss'].numpy(), 0, places=2)

    def test_fft_loss(self):
        for spatial_size in 2**torch.arange(8, 14):
            criterion = RFLoss(spatial_size=spatial_size, weight_fft=1)
            # create a pair of slices with different AM tones
            alpha = torch.vstack((torch.sin(torch.arange(spatial_size)), torch.zeros(spatial_size))).unsqueeze(0)
            omega = torch.vstack((torch.sin(torch.arange(spatial_size) * .1), torch.zeros(spatial_size))).unsqueeze(0)
            # fft_loss should be very high
            _, metrics = criterion(alpha, omega)
            self.assertGreater(metrics['fft_loss'], 0.95)
