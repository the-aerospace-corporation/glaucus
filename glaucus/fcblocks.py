# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import logging
import numpy as np

import torch
import lightning as pl

log = logging.getLogger(__name__)


class FullyConnected(pl.LightningModule):
    '''Sequential Layer Generator for 1D Fully Connected'''
    def __init__(self, size_in:int=512, size_out:int=128, steps:int=3,
                 quantize_in:bool=False, quantize_out:bool=False, use_dropout:bool=False) -> None:
        '''
        Smartly construct fully connected of arbitrary depth

        quantize_in means input will be dequantized from uint8
        quantize_out means output is quantized into uint8
        '''
        super().__init__()
        self.save_hyperparameters()
        self.size_in = size_in
        self.size_out = size_out
        self.steps = steps
        self.step_sizes = np.round(np.geomspace(self.size_in, self.size_out, self.steps+1)).astype(int)
        self.quantize_in = quantize_in
        self.quantize_out = quantize_out
        self.bn_mom = 1e-2 # better than torch default
        self.bn_eps = 1e-3 # better than torch default

        # deal with optional quantization
        if self.quantize_in or self.quantize_out:
            # 'fbgemm' is for servers, 'qnnpack' is for mobile
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
        if self.quantize_in:
            self._dequant_in = torch.quantization.DeQuantStub(qconfig=qconfig)
        if self.quantize_out:
            self._quant_out = torch.quantization.QuantStub(qconfig=qconfig)
        if use_dropout:
            self._dropout = torch.nn.Dropout(0.2)

        # define layers
        self._activ = torch.nn.SiLU(inplace=True)
        self._fc = self._make_fc(use_dropout)

        # layer summary
        summary = quantize_in * 'dequant,'
        for sdx in range(steps+1):
            summary += f'{self.step_sizes[sdx]}{"," * (sdx!=steps)}'
        summary += quantize_out * ',quant'
        log.info('FullyConnected(%s)', summary)

    def _make_fc(self, use_dropout=False):
        '''constructor for stepped architecture'''
        layers = []
        if self.quantize_in:
            layers.append(self._dequant_in)
        for sdx in range(self.steps):
            layers.append(torch.nn.Linear(self.step_sizes[sdx], self.step_sizes[sdx+1]))
            layers.append(torch.nn.BatchNorm1d(self.step_sizes[sdx+1], momentum=self.bn_mom, eps=self.bn_eps))
            layers.append(self._activ)
            if use_dropout and sdx != self.steps - 1:
                # do not add dropout on last layer
                layers.append(self._dropout)
        if self.quantize_out:
            layers.append(self._quant_out)
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x_hat = self._fc(x)
        return x_hat
