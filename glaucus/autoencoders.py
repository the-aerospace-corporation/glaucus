# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Autoencoder Model Definitions"""

import logging

import lightning as L
import torch
from madgrad import MADGRAD
from vector_quantize_pytorch import ResidualLFQ, ResidualVQ

from .fcblocks import FullyConnected
from .gblocks import DECODER_BLOCKS, ENCODER_BLOCKS, GlaucusNet
from .layers import (FreqDomain2TimeDomain, GaussianNoise, RMSNormalize,
                     TimeDomain2FreqDomain)
from .rfloss import RFLoss
from .seanet import SEANetEncoder, SEANetDecoder

log = logging.getLogger(__name__)

class GlaucusRVQVAE(L.LightningModule):
    """
    RF Vector Quantized Variational Autoencoder

    Differs from the original Glaucus AE in a few key ways:
    * Utilizes SEANet-style encoder/decoder pair
        - Can ingest variable length RF, but works best if slice length is a multiple of the combined compression ratios.
        - Includes optional LSTM to allow for historical information between slices to be utilized
    * Compresses via a residual-vector-quantized codebook architecture at the bottleneck, which quantizes encoded vectors into discrete "nearest approximations"

    Credit to Facebook's Encodec research: https://arxiv.org/abs/2210.13438
    """

    def __init__(
        self,
        spatial_size: int = 4096,
        compression_ratios: list = [8, 4, 2, 2],
        conv_filters: int = 32,
        lstm: int = 2,
        embed_dim: int = 128,
        quantizer: str = "vq",
        num_quantizers: int = 16,
        num_embed: int = 1024,
        add_noise: bool = False,
        data_format: str = "nl",
        **vq_kwargs,
    ) -> None:
        """
        spatial_size: int, default 4096, optional
            Length of RF input, only needed if adding training noise. Otherwise, you can input any length RF during inference.
        compression_ratios: list, default [8, 4, 4, 2]
            Downsampling/upsampling ratios of mirrored convolutional encoder and decoder. Output spatial size will be
            equal to input spatial size divided by all compression ratios, so make sure that the length of your input is
            at minimum the product of all compression_ratios.
        conv_filters: int, default 32
            Channel dimension of encoder/decoder pair. At each step in the convolutional network, doubles/halves for the
            encoder and decoder, respectively.
        lstm: int, default 2
            Number of LSTM layers to add at the end of encoder and at the beginning of the decoder. Set to 0 for no LSTM.
        embed_dim: int, default 128
            Dimension of latent space.
        num_quantizers: int, default 16
            Number of codebooks used to represent each embedded latent space value.
        num_embed: int, default 1024
            Number of total possible learned vectors per codebook. The total number of "bits" needed to represent the
            indices of each codebook is log2(num_embed)
        add_noise: bool, default False, optional
            Set to true if you need to add noise while training.
        data_format : str, default 'nl'
            Network normally consumes and produces complex-valued data represented as real-valued (NCL)
            but if data is complex-valued (NL) will add a transform layer during encode/decode.
        """
        super().__init__()

        self.save_hyperparameters()
        assert data_format in ["ncl", "nl"]
        assert num_embed <= 2**16, "num_embed/dictionary size too large"
        assert quantizer in ["vq", "lfq"]

        if num_embed <= 2**8:
            self._enc_idx_dtype = torch.uint8
        elif num_embed <= 2**16:
            self._enc_idx_dtype = torch.int16

        if quantizer == "vq":
            Quantizer = ResidualVQ
        elif quantizer == "lfq":
            Quantizer = ResidualLFQ

        self._noise_layer = GaussianNoise(spatial_size=spatial_size, min_snr_db=-10, max_snr_db=20)
        self.encoder = SEANetEncoder(channels=2, dimension=embed_dim, n_filters=conv_filters, ratios=compression_ratios, lstm=lstm)
        self.quantizer = Quantizer(codebook_size=num_embed, dim=embed_dim, num_quantizers=num_quantizers, **vq_kwargs)

        self.decoder = SEANetDecoder(channels=2, dimension=embed_dim, n_filters=conv_filters, ratios=compression_ratios, lstm=lstm)

    def forward(self, x):
        """
        Performs encoding and decoding in the same step. Use for verification purposes or training.
        """
        idx_or_enc, scale = self.encode(x)
        if self.training:
            enc = idx_or_enc.permute(0, 2, 1)
            z_emb, _, q_loss = self.quantizer(enc)
            q_loss = torch.mean(q_loss)
            x_hat = self.decode(idx_or_z_emb=z_emb, scale=scale)[..., : x.shape[-1]]
            return x_hat, q_loss
        else:
            x_hat = self.decode(idx_or_z_emb=idx_or_enc, scale=scale)[..., : x.shape[-1]]
            return x_hat

    def encode(self, x):
        """
        Encodes NCL or NL input into a quantized representation.
        During training, outputs quantized latent space. Otherwise,
        outputs codebook indices.

        Parameters
        ----------
        x: Tensor of shape NCL or NL

        Outputs
        -------
        enc_idx : Tensor
            Codebook indices of shape (N, spatial_out)
        OR
        z_emb : Tensor
            Quantized latent space representation of x (TRAINING ONLY)
        scale: Tensor
            Collection of scaling values to rescale output to correct values
        """

        if self.hparams.data_format == "ncl":
            x = torch.view_as_complex(x.swapaxes(-1, -2).contiguous())
        # RMS normalize before processing, save RMS scale factor to denormalize
        scale = x.norm(p=2, dim=-1, keepdim=True) / ((x.shape[-1]) ** 0.5)
        x = x / scale
        x = torch.view_as_real(x).swapaxes(-1, -2)
        # Add noise if training
        if self.hparams.add_noise:
            x, _ = self._noise_layer(x)
        z_emb = self.encoder(x)
        if self.training:
            return z_emb, scale

        z_emb = z_emb.permute(0, 2, 1)  # [N, C, L] -> [N, L, C]
        _, enc_idx, _ = self.quantizer(z_emb)
        enc_idx = enc_idx.type(self._enc_idx_dtype)

        return enc_idx, scale

    def decode(self, idx_or_z_emb, scale=None):
        """
        Takes in indices, generates quantized latent space, and reconstructs x_hat.
        During training, directly operates on quantized latent space instead.

        Parameters
        ----------
        idx_or_z_emb : Tensor
            Codebook indices of shape (N, spatial_out) OR quantized latent space representation of x (TRAINING ONLY)
        scale: Tensor, optional
            Optional scale factor to rescale output to proper amplitude. Omit if average power reconstruction is not needed.

        Outputs
        -------
        x_hat : Tensor
            Reconstructed version of x
        """
        if self.training:
            z_emb = idx_or_z_emb
        else:
            z_emb = self.quantizer.get_output_from_indices(idx_or_z_emb.type(torch.long))

        z_emb = z_emb.permute(0, 2, 1)  # [N, L, C] -> [N, C, L]
        x_hat = self.decoder(z_emb)

        if scale is not None:
            x_hat = x_hat * scale.view(-1, 1, 1)

        if self.hparams.data_format == "nl":
            x_hat = torch.view_as_complex(x_hat.swapaxes(-1, -2).contiguous())
        return x_hat

class GlaucusVAE(L.LightningModule):
    """
    RF Variational Autoencoder

    Different than the original Glaucus AE in a few key ways:
    * Instead of having a MLP as the bottleneck down, replace with 3 layers
        1) Encoder layer predicts mean
        2) Encoder layer predicts log(variance)
        3) Encoder samples this distribution and produces a tensor
    """

    def __init__(
        self,
        encoder_blocks: list = ENCODER_BLOCKS,
        decoder_blocks: list = DECODER_BLOCKS,
        domain: str = "time",
        width_coef: float = 1,
        depth_coef: float = 1,
        spatial_size: int = 4096,
        bottleneck_in: int = 512,
        bottleneck_latent: int = 512,
        bottleneck_out: int = 512,
        bottleneck_quantize: bool = False,
        data_format: str = "ncl",
        drop_connect_rate: float = 0.2,
        optimizer: str = "madgrad",
        lr: float = 1e-3,
    ) -> None:
        """
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
        """
        super().__init__()

        self.save_hyperparameters()
        assert domain in ["time", "freq"]
        assert data_format in ["ncl", "nl"]

        self._rms_norm = RMSNormalize(spatial_size=spatial_size)
        self._noise_layer = GaussianNoise(spatial_size=spatial_size)
        if domain == "freq":
            self._time2freq = TimeDomain2FreqDomain()
            self._freq2time = FreqDomain2TimeDomain()
        self.loss_function = RFLoss(spatial_size=spatial_size, data_format=data_format)

        self.encoder = GlaucusNet(encoder_blocks, mode="encoder", width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)
        self.encoder_mu = torch.nn.Linear(bottleneck_in, bottleneck_latent)
        self.encoder_var = torch.nn.Linear(bottleneck_in, bottleneck_latent)
        # projection back into decoder network, no activation
        self.decoder_fc = torch.nn.Linear(bottleneck_latent, bottleneck_out)
        self.decoder = GlaucusNet(decoder_blocks, mode="decoder", width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)

        if bottleneck_quantize:
            # 'fbgemm' is for servers, 'qnnpack' is for mobile
            qconfig = torch.quantization.get_default_qconfig("fbgemm")
            self._quant = torch.quantization.QuantStub(qconfig=qconfig)
            self._dequant = torch.quantization.DeQuantStub(qconfig=qconfig)

        ### End Extras
        optimizer_map = {"adam": torch.optim.Adam, "madgrad": MADGRAD}
        self.optimizer = optimizer_map[optimizer]

        log.info(f"GlaucusVAE({domain})")

    def forward(self, x):
        z_emb, p_z, q_z = self.encode(x)
        x_hat = self.decode(z_emb)
        return x_hat, z_emb, p_z, q_z

    def encode(self, x):
        """normalize, add noise if training, and reduce to latent domain"""
        if self.hparams.data_format == "nl":
            x = torch.view_as_real(x).swapaxes(-1, -2)
        x = self._rms_norm(x)
        if self.hparams.domain == "freq":
            # convert to frequency domain
            x = self._time2freq(x)
        feat_ef = self.encoder(x)
        z_mu = self.encoder_mu(feat_ef)
        # the output of encoder_var is actually log_var and we convert to var this way
        # trick from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        z_var = torch.nn.functional.softplus(self.encoder_var(feat_ef))

        p_z, q_z, z_emb = self.sample_pqz(z_mu, z_var)
        if self.hparams.bottleneck_quantize:
            z_emb = self._quant(z_emb)

        return z_emb, p_z, q_z

    def sample_pqz(self, mu, var):
        """
        Sample the mean and variance

        Eps is added here to prevent collapsing behavior, without the var would
        sometimes vanish and cause err. Trick from
        https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        """
        q_z = torch.distributions.Normal(mu, var + 1e-5)
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(var))
        z = q_z.rsample()
        return p_z, q_z, z

    def vae_loss(self, p, q, kl_scale: float = 0.05):
        """
        Parameters
        ----------
        kl_scale : float
            Unscaled VAE Loss was around 40 during initial testing.
            Perhaps make 0.1 default as this value neared 0.4 (unscaled) during training.
        """
        kl = torch.distributions.kl_divergence(q, p)
        return kl.mean() * kl_scale

    def decode(self, z_emb):
        """return from latent domain to complex RF"""
        if self.hparams.bottleneck_quantize:
            z_emb = self._dequant(z_emb)
        feat_fc = self.decoder_fc(z_emb)
        x_hat = self.decoder(feat_fc)
        if self.hparams.domain == "freq":
            # convert back to time domain
            x_hat = self._freq2time(x_hat)
        if self.hparams.data_format == "nl":
            x_hat = torch.view_as_complex(x_hat.swapaxes(-1, -2).contiguous())
        return x_hat

    def step(self, batch, batch_idx):
        x, metadata = batch
        x_hat, _, p, q = self.forward(x)
        loss, metrics = self.loss_function(x_hat, x)
        metrics["rfloss"] = loss
        loss_vae = self.vae_loss(p, q)
        loss += loss_vae
        metrics["vae_loss"] = loss_vae
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, metrics = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in metrics.items()}, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        return optimizer


class GlaucusAE(L.LightningModule):
    """RF Autoencoder constructed with network of GBlocks."""

    def __init__(
        self,
        encoder_blocks: list = ENCODER_BLOCKS,
        decoder_blocks: list = DECODER_BLOCKS,
        domain: str = "time",
        width_coef: float = 1,
        depth_coef: float = 1,
        spatial_size: int = 4096,
        bottleneck_in: int = 512,
        bottleneck_latent: int = 512,
        bottleneck_out: int = 512,
        bottleneck_steps: int = 1,
        bottleneck_quantize: bool = False,
        data_format: str = "ncl",
        drop_connect_rate: float = 0.2,
        optimizer: str = "madgrad",
        lr: float = 1e-3,
    ) -> None:
        """
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
        """
        super().__init__()

        self.save_hyperparameters()
        assert domain in ["time", "freq"]
        assert data_format in ["ncl", "nl"]
        self.domain = domain
        self.data_format = data_format

        self._rms_norm = RMSNormalize(spatial_size=spatial_size)
        self._noise_layer = GaussianNoise(spatial_size=spatial_size)
        if self.domain == "freq":
            self._time2freq = TimeDomain2FreqDomain()
            self._freq2time = FreqDomain2TimeDomain()
        self.loss_function = RFLoss(spatial_size=spatial_size, data_format=data_format)
        self.encoder = GlaucusNet(encoder_blocks, mode="encoder", width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)
        self.fc_encoder = FullyConnected(size_in=bottleneck_in, size_out=bottleneck_latent, quantize_out=bottleneck_quantize, steps=bottleneck_steps)
        self.fc_decoder = FullyConnected(size_in=bottleneck_latent, size_out=bottleneck_out, quantize_in=bottleneck_quantize, steps=bottleneck_steps)
        self.decoder = GlaucusNet(decoder_blocks, mode="decoder", width_coef=width_coef, depth_coef=depth_coef, drop_connect_rate=drop_connect_rate)
        optimizer_map = {"adam": torch.optim.Adam, "madgrad": MADGRAD}
        self.optimizer = optimizer_map[optimizer]

        log.info("GlaucusAE(%s)", self.domain)

    def forward(self, x):
        x_enc = self.encode(x)
        x_hat = self.decode(x_enc)
        return x_hat, x_enc

    def encode(self, x):
        """normalize, add noise if training, and reduce to latent domain"""
        if self.data_format == "nl":
            x = torch.view_as_real(x).swapaxes(-1, -2)
        x = self._rms_norm(x)
        # if training a denoising autoencoder is desired, enable following layer
        # x, _ = self._noise_layer(x)
        if self.domain == "freq":
            # convert to frequency domain
            x = self._time2freq(x)
        feats = self.encoder(x)
        x_enc = self.fc_encoder(feats)
        return x_enc

    def decode(self, x_enc):
        """return from latent domain to complex RF"""
        feats = self.fc_decoder(x_enc)
        x_hat = self.decoder(feats)
        if self.domain == "freq":
            # convert back to time domain
            x_hat = self._freq2time(x_hat)
        if self.data_format == "nl":
            x_hat = torch.view_as_complex(x_hat.swapaxes(-1, -2).contiguous())
        return x_hat

    def step(self, batch, batch_idx):
        x, metadata = batch
        x_hat, _ = self.forward(x)
        loss, metrics = self.loss_function(x_hat, x)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, _ = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
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
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        return optimizer


class FullyConnectedAE(L.LightningModule):
    """RF Autoencoder constructed with fully connected layers."""

    def __init__(
        self,
        spatial_size: int = 4096,
        latent_dim: int = 512,
        lr: float = 1e-3,
        steps: int = 3,
        bottleneck_quantize: bool = False,
        domain: str = "time",
        data_format: str = "ncl",
        optimizer: str = "madgrad",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        assert domain in ["time", "freq"]
        assert data_format in ["ncl", "nl"]
        self.domain = domain
        self.data_format = data_format

        self.latent_dim = latent_dim
        self.io_dim = spatial_size * 2
        self.steps = steps

        self._rms_norm = RMSNormalize(spatial_size=spatial_size)
        self._noise_layer = GaussianNoise(spatial_size=spatial_size)
        if self.domain == "freq":
            self._time2freq = TimeDomain2FreqDomain()
            self._freq2time = FreqDomain2TimeDomain()
        self.loss_function = RFLoss(spatial_size=spatial_size, data_format=data_format)

        optimizer_map = {"adam": torch.optim.Adam, "madgrad": MADGRAD}
        self.optimizer = optimizer_map[optimizer]

        self.encoder = FullyConnected(size_in=self.io_dim, size_out=self.latent_dim, quantize_out=bottleneck_quantize, steps=steps, use_dropout=True)
        self.decoder = FullyConnected(size_in=self.latent_dim, size_out=self.io_dim, quantize_in=bottleneck_quantize, steps=steps, use_dropout=True)
        # replace final activation with Tanh
        self.decoder._fc[-1] = torch.nn.Tanh()

        log.info("FullyConnectedAE(%s)", self.domain)

    def forward(self, x):
        x_enc = self.encode(x)
        x_hat = self.decode(x_enc)
        return x_hat, x_enc

    def encode(self, x):
        """normalize, add noise if training, and reduce to latent domain"""
        if self.data_format == "nl":
            x = torch.view_as_real(x).swapaxes(-1, -2)
        x = self._rms_norm(x)
        x, _ = self._noise_layer(x)
        if self.domain == "freq":
            # convert to frequency domain
            x = self._time2freq(x)
        # NCL to NL
        x = x.flatten(start_dim=1)
        x_enc = self.encoder(x)
        return x_enc

    def decode(self, x_enc):
        """return from latent domain to complex RF"""
        x_hat = self.decoder(x_enc)
        # NL to NCL
        x_hat = x_hat.view(x_hat.size(0), 2, -1)
        if self.domain == "freq":
            # convert back to time domain
            x_hat = self._freq2time(x_hat)
        if self.data_format == "nl":
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
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()})
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer
