#!/usr/bin/env python3
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

with open('README.md') as derp:
    long_description = derp.read()

setup(
    name='glaucus',
    version=version,
    author='Kyle Logue',
    author_email='kyle.logue@aero.org',
    url='https://github.com/the-aerospace-corporation/glaucus',
    license='GNU Lesser General Public License v3 or later (LGPLv3+)',
    test_suite='tests',
    packages=['glaucus'],
    description='Complex-valued encoder, decoder, and loss for RF DSP in PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',        # basic ML framework
        'lightning',    # extensions for PyTorch
        'madgrad',      # our favorite optimizer
        'hypothesis',   # best unit testing
    ],
)
