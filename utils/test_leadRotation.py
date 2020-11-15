#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle

import numpy as np
import scipy
from matplotlib import pyplot as plt

from dependencies import ROOTDIR, FILEDIR
from utils.HelperFunctions import Output, Configuration
from utils import leadRotation

inputfolder = os.path.join(FILEDIR, 'NIFTI')
side = 'left'

leadRotation.PrepareData().getData(subj='subj1', inputfolder=inputfolder, side=side)
