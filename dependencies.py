#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

# ====================    General settings for GUIs/functions within ./utils   ====================
sys.path.append('/media/storage/cDBS')
ROOTDIR = os.path.dirname(os.path.realpath(__file__))
FILEDIR = os.path.join(ROOTDIR, 'data')
GITHUB = 'https://github.com/dpedrosac/cDBS'
ITKSNAPv = '3.6.0'

# ====================    Settings for different leads available   ====================
lead_settings = {'Boston Vercise Directional': {'markerposition': 10.25, 'leadspacing': 2},
                 'St Jude 6172': {'markerposition': 9, 'leadspacing': 2},
                 'St Jude 6173': {'markerposition': 12, 'leadspacing': 3}}

# ====================    Settings for lead visualisation after estimation   ====================
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

