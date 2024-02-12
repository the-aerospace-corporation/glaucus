'''utilities'''
# Copyright 2023 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import re


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
