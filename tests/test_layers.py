'''ensure layers are working'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest
import numpy as np

import torch

from glaucus import (
    TimeDomain2FreqDomain,
    FreqDomain2TimeDomain,
    RMSNormalize,
    GaussianNoise,
    RFLoss,
)


class TestDomainTransforms(unittest.TestCase):
    def test_roundtrip(self):
        '''
        Time -> Freq -> Time should yield identical results
        Freq -> Time -> Freq should yield identical results
        '''
        batch_size = np.random.randint(0, 64)
        spatial_size = 2**np.random.randint(8, 16)
        original = torch.rand(batch_size, 2, spatial_size)
        layer_f2t = FreqDomain2TimeDomain()
        layer_t2f = TimeDomain2FreqDomain()
        roundtrip_ftf = layer_f2t(layer_t2f(original))
        roundtrip_tft = layer_t2f(layer_f2t(original))
        self.assertTrue(torch.allclose(roundtrip_ftf, original, atol=1e-5))
        self.assertTrue(torch.allclose(roundtrip_tft, original, atol=1e-5))


class TestNormalization(unittest.TestCase):
    def test_rms_normalize(self):
        '''
        Tests the RMSNormalize layer to ensure the layer is normalizing inputs to RMS.
        '''
        batch_size = np.random.randint(1, 64)
        spatial_size = 2**np.random.randint(8, 16)
        # generate batches with different means and stdevs
        means = np.geomspace(1e-2, 1e3, 6) * (np.random.randint(0, 2, size=6) * 2 - 1)
        stdevs = np.geomspace(1e-2, 1e8, 4)
        layer = RMSNormalize(spatial_size=spatial_size)
        for mean in means:
            for stdev in stdevs:
                x_ncl = mean + torch.randn(batch_size, 2, spatial_size) * stdev
                y_ncl = layer(x_ncl)
                # convert from reals to complex for analysis
                y_nl = torch.view_as_complex(y_ncl.swapaxes(-1, -2).contiguous())
                # recall that to calculate true RMS you can either use this, norm, or vdot
                y_rms = torch.sqrt(torch.mean(torch.square(torch.abs(y_nl)), axis=-1))
                self.assertTrue(torch.allclose(y_rms, torch.ones(batch_size), rtol=1e-2))


class TestGaussianNoise(unittest.TestCase):
    def test_skip_on_eval(self):
        '''
        When self.training == True (before eval) noise will be added with this layer.
        Otherwise it will just return the same input.
        '''
        noise_layer = GaussianNoise(spatial_size=64)
        alpha = torch.randn(1, 2, 64)
        omega, _ = noise_layer(alpha)
        self.assertFalse(torch.equal(alpha, omega))
        noise_layer.eval()
        omega, _ = noise_layer(alpha)
        self.assertTrue(torch.equal(alpha, omega))

    def test_snr_ranges(self):
        '''lower the SNR, lower the relationship to original signal'''
        alpha = torch.randn(1, 2, 64)
        rfloss = RFLoss(weight_spec=0)
        for min_snr_db in np.arange(-10, 15, 5):
            high_noise_layer = GaussianNoise(
                spatial_size=64,
                min_snr_db=min_snr_db,
                max_snr_db=min_snr_db+1)
            low_noise_layer = GaussianNoise(
                spatial_size=64,
                min_snr_db=min_snr_db+5,
                max_snr_db=min_snr_db+6)
            omega_high, _ = high_noise_layer(alpha)
            omega_low, _ = low_noise_layer(alpha)
            self.assertLess(
                rfloss(alpha, omega_low)[0],
                rfloss(alpha, omega_high)[0]
            )
