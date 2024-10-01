# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Utilities"""

import copy
import re
from math import gcd

import torch
from cbitstruct import pack, unpack


def adapt_glaucus_quantized_weights(state_dict: dict) -> dict:
    """
    The pretrained Glaucus models have a quantization layer that shifts the
    encoder list positions, so if we create a model w/o quantization we have to
    shift those layers slightly to make the pretrained model work.

    This function decrements the position of the decoder layers in the state
    dict to allow loading from a pre-trained model that was quantization aware.

    ie: `fc_decoder._fc.1.weight` becomes `fc_decoder._fc.0.weight`

    There will be extra layers remaining, but we can discard them by loading
    with `strict=False`. See the README for an example.

    Parameters
    ----------
    state_dict : dict
        Torch state dictionary including quantization layers.

    Returns
    -------
    new_state_dict : dict
        State dictionary without quantization layers.
    """
    new_state_dict = copy.deepcopy(state_dict)

    pattern = r"(fc_decoder._fc.)(\d+)(\.\w+)"  # regex pattern

    for key, value in state_dict.items():
        match = re.match(pattern, key)
        if match:
            extracted_int = int(match.group(2))
            new_key = f"{match.group(1)}{extracted_int-1}{match.group(3)}"
            new_state_dict[new_key] = value
    return new_state_dict


def lcm(alpha: int, beta: int) -> int:
    """
    Lowest Common Multiple
    note: this is built-in to math stdlib in python3.9+
    """
    return abs(alpha * beta) // gcd(alpha, beta)


def pack_tensor(ray: torch.Tensor, width: int = 10) -> bytes:
    """
    Pack 1D int tensor into bytes, keeping only width bits

    Raises TypeError if trying to pack values larger than width.
    """
    assert width < 16
    dtype = torch.uint8 if width <= 8 else torch.int16
    values_per_group = lcm(width, 8) // width
    assert ray.ndim == 1, "can only pack 1D arrays"
    assert len(ray) % values_per_group == 0, f"length of array must be multiple of {values_per_group}, is {len(ray)}"
    buffer = bytes()
    for group in ray.reshape(-1, values_per_group):
        buffer += pack(f"u{width}" * values_per_group, *group.tolist())
    return buffer


def unpack_tensor(buffer: bytes, width: int = 10) -> torch.Tensor:
    """unpack 1D tensor from bytes, keeping only width bits"""
    assert width < 16
    dtype = torch.uint8 if width <= 8 else torch.int16
    bytes_per_group = lcm(width, 8) // 8
    values_per_group = lcm(width, 8) // width
    assert len(buffer) % bytes_per_group == 0, f"number of bytes must be multiple of {bytes_per_group}, is {len(buffer)}"
    values = ()
    for gdx in range(len(buffer) // bytes_per_group):
        group = buffer[gdx * bytes_per_group : (gdx + 1) * bytes_per_group]
        values += unpack(f"u{width}" * values_per_group, group)
    return torch.tensor(values, dtype=dtype)
