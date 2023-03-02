'''custom layers for pytorch'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import typing
import torch
import lightning as pl


def wrap_ncl(func):
    '''
    decorator that allows the use of NCL (batchsize, 2, spatial_size) RF data
    with modules designed for complex RF NL (batchsize, spatial_size) RF data.
    '''
    def wrap(x_ncl, *args, **kwargs):
        x_nl = torch.view_as_complex(x_ncl.swapaxes(-1, -2).contiguous())
        y_nl = func(x_nl, *args, **kwargs)
        y_ncl = torch.view_as_real(y_nl).swapaxes(-1, -2)
        return y_ncl
    return wrap


class RMSNormalizeIQ(pl.LightningModule):
    '''
    When consuming RF, ensure the waveform has uniform scale to regularize input to our architecture.
    Expects Complex-Valued NL format (batchsize, spatial_size).

    Best Method
    -----------
    With numpy the [vdot]_ method is fastest, but some testing showed that using [norm]_ was best with pytorch.

    | Method   | CPU Time (ms) | GPU Time (µs) |
    |----------|---------------|---------------|
    |  [norm]_ |      10.1     |      203      |
    |  [vdot]_ |      13.3     |      386      |
    | [naive]_ |      14.3     |      462      |

    Tests were run on Threadripper 3990WX and single RTX2080. Input shape was (512, 8192).

    RMS Methods
    -----------
    All these methods find RMS amplitude equally.
    .. [norm] x.norm(p=2, dim=-1, keepdim=True)
    .. [vdot] torch.sqrt(torch.bmm(x.view(batch_size, 1, spatial_size), torch.conj(x.view(batch_size, spatial_size, 1))).squeeze(-1).real)
    .. [naive] torch.sqrt(torch.sum(torch.square(torch.abs(x)), axis=-1, keepdim=True))
    '''
    def __init__(self, spatial_size: int = 4096, bias: bool = False) -> None:
        super().__init__()
        self.bias = bias
        # recall that if something is a parameter it will be learned, and registered as a parameter automatically.
        # dis-similarly if something is a tensor it will not be learned, and needs to be a registered as a buffer to appear in model params
        # self.register_buffer('scale', 1 / torch.sqrt(torch.tensor(spatial_size, dtype=torch.float)))
        # self.scale = torch.nn.Parameter(1 / torch.sqrt(torch.tensor(spatial_size, dtype=torch.float)), requires_grad=False)
        self.scale = 1 / torch.sqrt(torch.tensor(spatial_size, dtype=torch.float))
        if bias:
            self.offset = torch.nn.Parameter(torch.zeros(spatial_size, dtype=torch.float))

    def forward(self, x):
        '''
        Returns the original complex signal scaled appropriately.
        '''
        rms = x.norm(p=2, dim=-1, keepdim=True) * self.scale
        x_rms = x / rms
        if self.bias:
            x_rms += self.offset
        return x_rms


class RMSNormalize(RMSNormalizeIQ):
    '''Inherits RMSNormalize, but expects real-valued NCL format (batchsize, 2, spatial_size)'''
    def __init__(self, spatial_size: int = 4096, bias: bool = False):
        super().__init__(spatial_size, bias)
        self.forward = wrap_ncl(self.forward)


class TimeDomain2FreqDomain(pl.LightningModule):
    '''
    Convert Time Domain RF to Freq Domain RF.
    Expects NCL format (batchsize, 2, spatial_size).
    '''
    def __init__(self, fft_shift: bool = True, norm: typing.Optional[str] = None):
        super().__init__()
        if norm and norm not in ['forward', 'backward', 'ortho']:
            raise ValueError(f"Invalid normalization option: {norm}")
        self.fft_shift = fft_shift
        self.norm = norm

    def forward(self, x):
        '''
        fftshift is not strictly required, but keeps features spatially close for potential convolutional layers.
        '''
        x = torch.view_as_complex(x.swapaxes(-1, -2).contiguous())
        x = torch.fft.fft(x, dim=-1, norm=self.norm)
        if self.fft_shift:
            x = torch.fft.fftshift(x, dim=-1)
        x = torch.view_as_real(x).swapaxes(-1, -2)
        return x


class FreqDomain2TimeDomain(pl.LightningModule):
    '''
    Convert Freq Domain RF to Time Domain RF.
    Expects NCL format (batchsize, 2, spatial_size).
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        fftshift is not strictly required, but keeps features spatially close for potential convolutional layers.
        '''
        x = torch.view_as_complex(x.swapaxes(-1, -2).contiguous())
        x = torch.view_as_real(torch.fft.ifft(torch.fft.fftshift(x, dim=-1))).swapaxes(-1, -2)
        return x


class DropConnect(pl.LightningModule):
    '''
    drop connections between blocks (alternative to dropout)

    The intended use of dropconnect is within residual connections like:
    `out = in + dropconnect(layer(in))`

    Some implementations utilize activation after this step, but we have not.

    Parameters
    ----------
    inputs : torch.Tensor
        Of shape (batchsize, ...).
    drop_connect_rate : float
        Probability of drop connections (0-1).
    training : bool
        Only do this if we are in `training` mode.

    References
    ----------
    [1] http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf
    [2] https://github.com/tensorflow/tpu/blob/cd433314cc6f38c10a23f1d607a35ba422c8f967/models/official/efficientnet/utils.py#L146
    '''
    def __init__(self, drop_connect_rate:float=0.2):
        super().__init__()
        assert 0 <= drop_connect_rate <= 1, 'drop_connect_rate must be in range of [0, 1]'
        self.survival_rate = 1 - drop_connect_rate

    def __repr__(self):
        ''' This makes things look nice when you print the architecture.'''
        return f'{type(self).__qualname__}(survival_rate={self.survival_rate:.2f})'

    def forward(self, x):
        if not self.training:
            # only drop connections during training
            return x
        elif self.survival_rate == 1:
            return x
        else:
            batchsize = x.shape[0]
            # create binary tensor mask
            random_tensor = self.survival_rate + torch.rand((batchsize,) + (1,)*(x.dim()-1), dtype=x.dtype, device=x.device)
            binary_tensor = torch.floor(random_tensor)
            # Unlike canonical method to multiply survival_rate at test time, here we
            # divide survival_rate at training time, such that no addition compute is needed at test time.
            return x / self.survival_rate * binary_tensor


class GaussianNoise(torch.nn.Module):
    '''
    Add gaussian noise to RF.
    Expects NCL format like (batchsize, 2, spatial_size).

    Input should be RMS normalized.
    Returns RMS normalized output.
    '''
    def __init__(self, spatial_size:int=4096, min_snr_db:float=-3, max_snr_db: float = 20):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self._rms_norm = RMSNormalize(spatial_size=spatial_size)

    def forward(self, x):
        if self.training:
            snr_db = torch.rand(1, device=x.device)[0] * (self.max_snr_db - self.min_snr_db) + self.min_snr_db
            snr = 10**(snr_db/10)
            # unit noise already RMS normalized
            unit_noise = torch.randn(x.size(), device=x.device, dtype=x.dtype) * 0.7071067811865476
            # scale signal and noise
            x = unit_noise + x * snr
            # as much as I want to use something like this, it's not quite the same as normalizing again
            # unit_noise / (snr + 1) + x * snr / (snr + 1)
            x = self._rms_norm(x)

        if self.training:
            x = x, snr_db
        else:
            x = x, None
        return x
