# Copyright 2024 The Aerospace Corporation
# This file is a part of Glaucus
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import unittest

import torch
from hypothesis import given
from hypothesis import strategies as st

from glaucus import lcm, pack_tensor, unpack_tensor


class TestPackUnpack(unittest.TestCase):
    @given(
        width=st.integers(min_value=1, max_value=15),
        count=st.integers(min_value=1, max_value=64),
    )
    def test_roundtrip_from_bytes(self, width, count):
        """Roundtrip unpack -> pack"""
        # some = torch.arange(4)
        # buffer = pack_tensor(some, width=10)
        min_bytes = lcm(width, 8) // 8
        num_bytes = min_bytes * count
        alpha_buffer = os.urandom(num_bytes)
        some = unpack_tensor(alpha_buffer, width=width)
        omega_buffer = pack_tensor(some, width=width)
        self.assertEqual(alpha_buffer, omega_buffer)

    @given(st.integers(min_value=3, max_value=15))
    def test_widths(self, width):
        """
        Ensure all valid bit widths work correctly
        """
        some_in = torch.arange(8, dtype=torch.uint8)
        buffer = pack_tensor(some_in, width)
        some_out = unpack_tensor(buffer, width)
        self.assertTrue(torch.all(some_in == some_out))
