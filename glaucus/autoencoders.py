# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging

import torch
import lightning as pl
from madgrad import MADGRAD

from .gblocks import GlaucusNet, ENCODER_BLOCKS, DECODER_BLOCKS
from .fcblocks import FullyConnected
from .layers import RMSNormalize, GaussianNoise, TimeDomain2FreqDomain, FreqDomain2TimeDomain
from .rfloss import RFLoss

log = logging.getLogger(__name__)


class GlaucusAE(pl.LightningModule):
    '''RF Autoencoder constructed with network of GBlocks.'''
    def __init__(self, encoder_blocks=ENCODER_BLOCKS, decoder_blocks=DECODER_BLOCKS,
            domain:str='time', width_coef:float=1, depth_coef:float=1, spatial_size:int=4096,
            bottleneck_in:int=512, bottleneck_latent:int=512, bottleneck_out:int=512,
            bottleneck_steps:int=1, bottleneck_quantize:bool=False, data_format:str='ncl',
            drop_connect_rate:float=0.2, optimizer:str='madgrad', lr:float=1e-3,
            ) -> None:
        '''
        encoder_blocks : list of namedtuple, optional
            Parameters to define sequential neural net architecture. Note that encoder output shape should be compatible
            with bottleneck parameters.
        decoder_blocks : list of namedtuple, optional
            See encoder blocks.
        domain : str, default time
            Force the encoder/decoder to operate in either time or freq domain.
        width_coef : float, default 1
            Coefficient to scale the width of the encoder/decoder blocks providing more channels for a given architecture.
        depth_coef : float, default 1
            Coefficient to scale the depth of the encoder/decoder blocks increasing complexity (and hopefully capacity).
        spatial_size : int, default 4096
            This network consumes RF data in NCL format. In order to properly do normalization properly and keep losses
            consistent we need to know the input length up front.
        bottleneck_in : int, default 512
            After feature extraction we enter a bottleneck layer. The input size of this layer should be equal to the
            output size of the encoder defined in `encoder_blocks`.
        bottleneck_latent : int, default 512
            The smallest point of the bottleneck that defines the size of our latent space.
        bottleneck_out : int, default 512
            The size of the bottleneck output should be equal to the input size of the decoder defined within
            `decoder_blocks`.
        bottleneck_step : int, default 1
            Within the bottleneck this determines how many steps down/up to/from the latent space.
        bottleneck_quantize : bool, default False
            This option adds a quant/dequant step in the latent space for maximum compression.
        drop_connect_rate : float, optional
            Drop connect is a generalization of dropout that allows better regularization for deeper networks.
            As implemented the rate will scale from 0 for the first layer up to the provided drop_connect_rate for the
            final layer. `drop_connect_rate` is the inverse of `survival_rate`.
            Very large models can generalize training with values up to 0.5, but in Dec 2021
            a comparison between using max of 0.2 and a scaled max value (0.2 to 0.5) depending on model size
            determined the larger rates only extended training time and didn't generally improve. Stick with 0.2.
        optimizer : string, default madgrad
            Currently support either `madgrad` or `adam` optimizers.
        lr : float, default 1e-3
            Learning Rate. Experiments from Dec 2021 to Mar 2022 yielded good values in range (1e-3, 1e-2).
        data_format : str, default 'ncl'
            Network normally consumes and produces complex-valued data represented as real-valued (NCL)
            but if data is complex-valued (NL) will add a transform layer during encode/decode.
        '''
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        assert domain in ['time', 'freq']
        assert data_format in ['ncl', 'nl']
        self.domain = domain
        self.data_format = data_format

        self._rms_norm = RMSNormalize(spatial_size=spatial_size)
        self._noise_layer = GaussianNoise(spatial_size=spatial_size)
        if self.domain == 'freq':
            self._time2freq = TimeDomain2FreqDomain()
            self._freq2time = FreqDomain2TimeDomain()
        self.loss_function = RFLoss(spatial_size=spatial_size, data_format=data_format)
        self.encoder = GlaucusNet(
            encoder_blocks, mode='encoder', width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)
        self.fc_encoder = FullyConnected(
            size_in=bottleneck_in, size_out=bottleneck_latent, quantize_out=bottleneck_quantize, steps=bottleneck_steps)
        self.fc_decoder = FullyConnected(
            size_in=bottleneck_latent, size_out=bottleneck_out, quantize_in=bottleneck_quantize, steps=bottleneck_steps)
        self.decoder = GlaucusNet(
            decoder_blocks, mode='decoder',
            width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)
        optimizer_map = {'adam': torch.optim.Adam, 'madgrad': MADGRAD}
        self.optimizer = optimizer_map[optimizer]

        log.info('GlaucusAE(%s)', self.domain)

    def forward(self, x):
        x_enc = self.encode(x)
        x_hat = self.decode(x_enc)
        return x_hat, x_enc

    def encode(self, x):
        '''normalize, add noise if training, and reduce to latent domain'''
        if self.data_format == 'nl':
            x = torch.view_as_real(x).swapaxes(-1,-2)
        x = self._rms_norm(x)
        x, _ = self._noise_layer(x)
        if self.domain == 'freq':
            # convert to frequency domain
            x = self._time2freq(x)
        feats = self.encoder(x)
        x_enc = self.fc_encoder(feats)
        return x_enc

    def decode(self, x_enc):
        '''return from latent domain to complex RF'''
        feats = self.fc_decoder(x_enc)
        x_hat = self.decoder(feats)
        if self.domain == 'freq':
            # convert back to time domain
            x_hat = self._freq2time(x_hat)
        if self.data_format == 'nl':
            x_hat = torch.view_as_complex(x_hat.swapaxes(-1,-2).contiguous())
        return x_hat

    def step(self, batch, batch_idx):
        x, metadata = batch
        x_hat, _ = self.forward(x)
        loss, metrics = self.loss_function(x_hat, x)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in metrics.items()}, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer


class FullyConnectedAE(pl.LightningModule):
    '''RF Autoencoder constructed with fully connected layers.'''
    def __init__(self,
        spatial_size:int=4096, latent_dim:int=512, lr:float=1e-3, steps:int=3, bottleneck_quantize:bool=False,
        domain:str='time', data_format:str='ncl', optimizer:str='madgrad') -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        assert domain in ['time', 'freq']
        assert data_format in ['ncl', 'nl']
        self.domain = domain
        self.data_format = data_format

        self.latent_dim = latent_dim
        self.io_dim = spatial_size * 2
        self.steps = steps

        self._rms_norm = RMSNormalize(spatial_size=spatial_size)
        self._noise_layer = GaussianNoise(spatial_size=spatial_size)
        if self.domain == 'freq':
            self._time2freq = TimeDomain2FreqDomain()
            self._freq2time = FreqDomain2TimeDomain()
        self.loss_function = RFLoss(spatial_size=spatial_size, data_format=data_format)

        optimizer_map = {'adam': torch.optim.Adam, 'madgrad': MADGRAD}
        self.optimizer = optimizer_map[optimizer]

        self.encoder = FullyConnected(
            size_in=self.io_dim, size_out=self.latent_dim, quantize_out=bottleneck_quantize, steps=steps, use_dropout=True)
        self.decoder = FullyConnected(
            size_in=self.latent_dim, size_out=self.io_dim, quantize_in=bottleneck_quantize, steps=steps, use_dropout=True)
        # replace final activation with Tanh
        self.decoder._fc[-1] = torch.nn.Tanh()

        log.info('FullyConnectedAE(%s)', self.domain)

    def forward(self, x):
        x_enc = self.encode(x)
        x_hat = self.decode(x_enc)
        return x_hat, x_enc

    def encode(self, x):
        '''normalize, add noise if training, and reduce to latent domain'''
        if self.data_format == 'nl':
            x = torch.view_as_real(x).swapaxes(-1, -2)
        x = self._rms_norm(x)
        x, _ = self._noise_layer(x)
        if self.domain == 'freq':
            # convert to frequency domain
            x = self._time2freq(x)
        # NCL to NL
        x = x.flatten(start_dim=1)
        x_enc = self.encoder(x)
        return x_enc

    def decode(self, x_enc):
        '''return from latent domain to complex RF'''
        x_hat = self.decoder(x_enc)
        # NL to NCL
        x_hat = x_hat.view(x_hat.size(0), 2, -1)
        if self.domain == 'freq':
            # convert back to time domain
            x_hat = self._freq2time(x_hat)
        if self.data_format == 'nl':
            x_hat = torch.view_as_complex(x_hat.swapaxes(-1, -2).contiguous())
        return x_hat

    def step(self, batch, batch_idx):
        x, metadata = batch
        x_hat, _ = self.forward(x)
        loss, metrics = self.loss_function(x_hat, x)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx)
        # self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
