#!/usr/bin/env python3
#-*- coding: utf-8 -*-
'''
  ________.__
 /  _____/|  | _____   __ __   ____  __ __  ______
/   \  ___|  | \__  \ |  |  \_/ ___\|  |  \/  ___/
\    \_\  \  |__/ __ \|  |  /\  \___|  |  /\___ \
 \______  /____(____  /____/  \___  >____//____  >
        \/          \/            \/           \/
'''
import os
import re
from setuptools import setup

with open(os.path.join('glaucus', '__init__.py'), encoding='utf-8') as derp:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', derp.read()).group(1)

setup(
    name='glaucus',
    version=version,
    author='Kyle Logue',
    author_email='kyle.logue@aero.org',
    test_suite='tests',
    packages=['glaucus'],
    description='Complex-valued encoder, decoder, and loss for RF DSP in PyTorch.',
    long_description=__doc__,
    install_requires=[
        'torch',                # basic ML framework
        'pytorch_lightning',    # extensions for PyTorch
    ],
)
