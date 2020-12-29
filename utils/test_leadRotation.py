#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dependencies import FILEDIR
from utils import leadRotation

inputfolder = os.path.join(FILEDIR, 'NIFTI')
side = 'right'

leadRotation.PrepareData().getData(subj='subj3', input_folder=inputfolder, side=side)
