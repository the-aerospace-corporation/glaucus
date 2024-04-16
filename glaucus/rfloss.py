# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Radio Frequency Loss"""

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .layers import RMSNormalizeIQ


class RFLoss(_Loss):
    """Loss criterion for complex RF timeseries."""

    def __init__(
        self,
        spatial_size: int = 4096,
        weight_spec: float = 1,
        weight_xcor: float = 1,
        weight_align: float = 1,
        weight_env: float = 1,
        weight_fft: float = 0,
        data_format: str = "ncl",
        overlap_frac: int = 0,
        reduction: str = "mean",
    ) -> None:
        """
        This criterion was the result of a long experimentation phase and tries
        to balance the loss from each component of the calculation.

        Any weights that equal zero will skip computation.

        Parameters
        ----------
        spatial_size : int, default 4096
            The length of the input complex tensors. If omitted the weights between metrics will no longer be uniform.
        weight_spec : float, default 1
            Valuable metric that compares inputs with multiple spectrograms at
            various resolutions.
        weight_xcor : float, default 1
            Valuable metric that estimates real & imaginary time-domain error.
        weight_align: float, default 1
            Metric that emphasizes the importance of the DC alignment, a
            component of the full xcor.
        weight_env : float, default 1
            OK metric that ensure the scale of inputs w.r.t phase are similar
            over the observation period. In my experience ML autoencoders learn
            this quickly then do not improve.
        weight_fft : float, default 0
            This component disabled since it is simply a much worse version of
            the spectrogram loss.
        data_format : str, default 'ncl'
            Network normally consumes and produces complex-valued data
            represented as real-valued (NCL) but if data is complex-valued (NL)
            will add a transform layer during encode/decode.
        overlap_frac : float, default 0
            Amount of overlap to use when calculating spectrograms. If nonzero
            uses slower spectral method. Reccommend 0.25 to 0.5 if enabled.

        Similarity Metrics
        ------------------
        * Are the signals in phase? -> env_loss
        * Are the signals similar in magnitude? -> env_loss
        * Are the signals similar in time domain? -> xcor_loss
        * Are the signals aligned in time domain? -> align_loss
        * Are the signals similar in freq domain? -> spec_loss or fft_loss
        * Are the signals similar in 2D spectral representation? -> spec_loss

        Error between Complex Waveforms
        -------------------------------
        In order to implement mean complex error we can use:
            1) mean((abs(a)-abs(b))**2) # compare magnitude only
            2) mean(abs(a-b)**2) == mean((a.real-b.real)**2 +
               (a.imag-b.imag)**2) # compare phase & mag
            3) mean(a*a.conj() - b*b.conj()) # worse magnitude comparison

        Notes
        -----
        * Beware: Magnitude & Phase comparisons made within env_loss are
          un-normalized (on purpose)
        * Spectrograms computed at multiple resolutions on normalized (apple,
          banana). Where `blah = log2(len(ray_apple)`, and appropriate fftsizes
          are `[2**(blah-2), 2**(blah-1), 2**blah, 2**(blah+1), 2**(blah+2)]`.
          Keep both magnitude and complex portion of error; complex portion
          hard to figure in early epochs. Ensures frequency domain correlation
          will work at multiple scales.
        * Normalized Complex Correlation loss ensures time domain
          crosscorrelation will be correct
        """
        super().__init__(reduction=reduction)
        assert data_format in ["ncl", "nl"]
        assert 0 <= overlap_frac < 1, "fractional value is between [0, 1)"
        spatial_exponent = np.log2(spatial_size)
        if spatial_exponent % 1 != 0 and overlap_frac == 0:
            log.warning('RFLoss on spatial size 2**%.2f may miss some end samples.', spatial_exponent)

        self.fftsizes = 2 ** (np.arange(-2, 3, dtype=int) + int(spatial_exponent / 2))
        self.spec_scale = 2 / len(self.fftsizes)
        self.spatial_size = 2**spatial_exponent
        self.weight_xcor = weight_xcor
        self.weight_align = weight_align
        self.weight_spec = weight_spec
        self.weight_env = weight_env
        self.weight_fft = weight_fft
        self._rms = RMSNormalizeIQ(spatial_size=spatial_size)
        self.data_format = data_format

        self.overlap_frac = overlap_frac
        if overlap_frac > 0:
            self.overlaps = np.round(self.fftsizes * self.overlap_frac).astype(int)
            for fdx, fftsize in enumerate(self.fftsizes):
                self.register_buffer(f'window{fdx}', torch.hann_window(fftsize))
                # adjust scale so output is comparable with non-overlap version
                self.spec_scale = 1

    def __repr__(self):
        return "{}(fftsizes={}, {}weights=({:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}), {})".format(
            type(self).__name__,
            "[" + " ".join("{}".format(ffz) for ffz in self.fftsizes) + "]",
            f"overlap={self.overlap_frac}, " if self.overlap_frac else "",
            self.weight_xcor,
            self.weight_align,
            self.weight_spec,
            self.weight_env,
            self.weight_fft,
            self.data_format.upper(),
        )

    @staticmethod
    def spec_overlap(ray: Tensor, fftsize:int, overlap:int, window: Tensor):
        """
        create high-resolution spectrogram with overlap & windowing
        """

        ray_len = ray.size(-1)
        hop_len = fftsize - overlap
        nrows = (ray_len - overlap) // hop_len

        if ray.ndim == 1:
            spec = torch.empty((nrows, fftsize), dtype=torch.complex64, device=ray.device)
            for rdx in range(nrows):
                spec[rdx] = ray[rdx*hop_len:rdx*hop_len+fftsize]
        elif ray.ndim == 2:
            spec = torch.empty((ray.size(0), nrows, fftsize), dtype=torch.complex64, device=ray.device)
            for rdx in range(nrows):
                spec[:, rdx] = ray[:, rdx*hop_len:rdx*hop_len+fftsize]
        else:
            raise ValueError('Expected complex-valued NL or L')

        spec = torch.fft.fft(spec * window, n=fftsize)

        # note that spec is un-shifted and complex
        return spec


    @staticmethod
    def spec_turbo(ray: Tensor, fftsize: int):
        """
        *fast* low-resolution spectrogram with zero overlap or padding

        Parameters
        ----------
        ray : Tensor
            ray can be (batchsize, complex_1d_tensor) or (complex_1_tensor) shape.
        fftsize : int
            Determine # of columns in spectrogram.
            Should be like 2 ** (round(log2(ray_len)) // 2) for square pixels.

        Returns
        -------
        spec : torch.Tensor
            2d tensor with un-fftshifted spectrogram.
        """
        ray_len = ray.shape[-1]
        nrows = torch.div(ray_len, fftsize, rounding_mode="floor")
        # doing reshape this way loses a few samples if ray_len / fftsize is not an integer!
        spec = ray[..., 0 : fftsize * nrows].reshape((-1, nrows, fftsize))
        spec = torch.fft.fft(spec, n=fftsize)
        # note that spec is un-shifted and complex
        return spec

    def calc_spec_loss(self, apple_cplx: Tensor, banana_cplx: Tensor):
        """
        loss from various spectrogram sizes

        Returns
        -------
        loss : Tensor
            Value in range(0, 1).
            Will be 1 if completely uncorrelated.
            Will be 0 if completely correlated.
        """
        spec_loss = 0
        for fdx, fftsize in enumerate(self.fftsizes):
            if self.overlap_frac:
                overlap = self.overlaps[fdx]  # overlap amount different per fftsize
                window = getattr(self, f"window{fdx}")  # retrieve precomputed window
                spec_apple = self.spec_overlap(apple_cplx, fftsize=fftsize, overlap=overlap, window=window)
                spec_banana = self.spec_overlap(banana_cplx, fftsize=fftsize, overlap=overlap, window=window)
            else:
                spec_apple = self.spec_turbo(apple_cplx, fftsize=fftsize)
                spec_banana = self.spec_turbo(banana_cplx, fftsize=fftsize)
            # comparison does not include phase

            spec_loss += torch.mean((torch.abs(spec_apple) - torch.abs(spec_banana)) ** 2) / fftsize
        return spec_loss * self.spec_scale

    def calc_xcor_loss(self, apple_fft: Tensor, banana_fft: Tensor):
        """
        cross-correlation loss

        track correlation of both real & imag to reconstruct phase

        Returns
        -------
        xcor_loss : Tensor
            Measure of rough time alignment in range(0, 1).
            Will be 1 if completely uncorrelated.
            Will be 0 if completely correlated at any time offset within slice.
        align_loss : Tensor
            Measure of fine time alignment in range(0, 1).
            Will be 1 if perfectly aligned in time.
            Will be 0 if no relationship.
        """
        # Peak value of xcor will be close to 1 if same pulse in both slices
        corr = torch.abs(torch.fft.ifft(apple_fft * torch.conj(banana_fft))) / self.spatial_size
        xcor_loss = 1 - torch.mean(torch.max(corr, dim=-1).values)
        # DC component of correlation should be maximized if aligned
        align_loss = 1 - torch.mean(torch.abs(corr[..., 0]))
        return xcor_loss, align_loss

    def calc_env_loss(self, apple_cplx: Tensor, banana_cplx: Tensor):
        """
        Envelope Loss
        Computed before normalization!

        TODO: without normalization will be avgpower(a)**2 + avgpower(b)**2

        Returns
        -------
        loss : Tensor
            Loss in range(0, 2) if similar scale; in range(0, inf) if unbounded.
            Will be 4 if oppositely correlated
            Will be 2 if completely uncorrelated.
            Will be <1 if similar scale, likely floating around 0.5.
            Will be 0 if identical.
        """
        # div2 to be similar to MSELoss
        return torch.mean(torch.abs(apple_cplx - banana_cplx) ** 2) * 0.5

    def calc_fft_loss(self, apple_fft: Tensor, banana_fft: Tensor):
        """
        compute loss from fullsize fft

        Returns
        -------
        loss : Tensor
            Value in range(0, 1).
            Will be 1 if freq domain uncorrelated.
            Will be 0 if freq domain looks identical.
        """
        return torch.mean((torch.abs(apple_fft) - torch.abs(banana_fft)) ** 2) / (2 * self.spatial_size)

    def forward(self, apple: Tensor, banana: Tensor):
        """
        compute loss and metrics

        If initialized with data_format == ncl, assume complex data in ray is
        real-valued (batchsize, 2, len) to represent complex-valued data channels first
        benchmark: 4096 len is 14 ms using batch_size=64 (223μs per row)

        Parameters
        ----------
        apple : torch.Tensor
            Tensor with shape (batchsize, 2, spatial_size) that will be converted to torch.complex64.
        banana : torch.Tensor
            See apple.
        """
        if self.data_format == "ncl":
            apple_cplx = torch.view_as_complex(apple.swapaxes(-1, -2).contiguous())
            banana_cplx = torch.view_as_complex(banana.swapaxes(-1, -2).contiguous())
        else:
            apple_cplx = apple
            banana_cplx = banana

        metrics = {}
        total_loss = torch.tensor(0, device=apple.device, dtype=torch.float)
        if self.weight_env:
            env_loss = self.weight_env * self.calc_env_loss(apple_cplx, banana_cplx)
            metrics["env_loss"] = env_loss
            total_loss += env_loss
        # normalize for other loss considerations
        apple_cplx = self._rms(apple_cplx)
        banana_cplx = self._rms(banana_cplx)
        if self.weight_spec:
            spec_loss = self.weight_spec * self.calc_spec_loss(apple_cplx, banana_cplx)
            metrics["spec_loss"] = spec_loss
            total_loss += spec_loss
        if self.weight_xcor or self.weight_align or self.weight_fft:
            # observe freq domain if necessary
            apple_fft = torch.fft.fft(apple_cplx)
            banana_fft = torch.fft.fft(banana_cplx)
        if self.weight_xcor or self.weight_align:
            xcor_loss, align_loss = self.calc_xcor_loss(apple_fft, banana_fft)
            # weight * .25 to make similar magnitude to other components
            if self.weight_xcor:
                xcor_loss *= self.weight_xcor * 0.25
                metrics["xcor_loss"] = xcor_loss
                total_loss += xcor_loss
            if self.weight_align:
                align_loss *= self.weight_align * 0.25
                metrics["align_loss"] = align_loss
                total_loss += align_loss
        if self.weight_fft:
            fft_loss = self.weight_fft * self.calc_fft_loss(apple_fft, banana_fft)
            metrics["fft_loss"] = fft_loss
            total_loss += fft_loss

        metrics["loss"] = total_loss
        return total_loss, metrics
