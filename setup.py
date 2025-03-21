# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import setuptools


NAME='brainmaze-eeg'
DESCRIPTION='BrainMaze: Brain Electrophysiology, Behavior and Dynamics Analysis Toolbox'
LONG_DESCRIPTION=open('README.rst', encoding='utf-8').read()
EMAIL='mivalt.filip@mayo.edu'
AUTHOR='Filip Mivalt'
REQUIRES_PYTHON = '>=3.9.0'
URL="https://github.com/bnelair/brainmaze_eeg"
PACKAGES = setuptools.find_packages()

REQUIRED = []
# if requirements___.txt exists, use it to populate the REQUIRED list
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        REQUIRED = f.read().splitlines()


setuptools.setup(
    name=NAME,
    use_scm_version=True,
    setup_requires=['setuptools>=61', 'setuptools_scm'],

    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license="BSD-3-Clause",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',

    packages=setuptools.find_packages(exclude=["tests*"]),
    include_package_data=True,

    classifiers=[
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        'Operating System :: OS Independent',
        "License :: OSI Approved :: BSD License",
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
    ],
    python_requires=REQUIRES_PYTHON,
    install_requires =REQUIRED,
)






