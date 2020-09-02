#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A setuptools-based setup module for the the Imaging Toolbox which is supposed to analyse imaging data from DBS
patients"""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import setuptools.command.build_ext
import os
import subprocess

ROOTDIR = os.path.dirname(os.path.realpath(__file__))
version = "0.1.1"

class CMakeBuild(build_ext):
    def run(self):
        ## Find or Configure ANTs ##
        print("\t ... to set up the required directories")
        subprocess.run(['./scripts/configure_cDBS.sh'], cwd=ROOTDIR)

setup(
    name='cDBS',
    version=version,
    description='This projects intends to provide a toolbox for analysing imaging data from patients receiving DBS '
                'electrodes. Particularly, the preoperative MRI scans and postoperative CT scans are used to visualise'
                'the location of the electrodes.',
    # TODO, add: long_description='long_description',  # Optional
    url='https://github.com/dpedrosac/analysis-myoDVS',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Private',
        'Topic :: Software Development :: Data recording',
        'License :: MIT License',
        'Programming Language :: Python :: 3.6',
        ],
    cmdclass={"folders": CMakeBuild},
    packages=find_packages(),
    python_requires='>=2.7,!=3.0.*,!=3.1.*',
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'PyYAML',
        'antspyx'
    ],
    scripts=[
            # 'preprocess/prepare_data.py',
            # 'analysis/analyse_data.py',
           ],
    project_urls={
        'Bug Reports': 'https://github.com/dpedrosac/ImagingToolbox/issues',
        'Funding': 'https://donate.pypi.org',
        'Say Thanks!': '',
        'Source': 'https://github.com/dpedrosac/cDBS/',
        },
)
