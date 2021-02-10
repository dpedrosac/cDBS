#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# ====================    General settings for GUIs/functions within ./utils   ====================
ROOTDIR = os.path.dirname(os.path.realpath(__file__))
FILEDIR = os.path.join(ROOTDIR, 'data')
GITHUB = 'https://github.com/dpedrosac/cDBS'
ITKSNAPv = '3.6.0'

# ====================    Settings for lead visualisation after estimation   ====================
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}