#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from PyQt5.QtWidgets import QFileDialog
from dependencies import ROOTDIR, ITKSNAPv
from utils.HelperFunctions import Configuration


class FileLocation:
    """in this class, the default folders for the additional toolboxes can be stored. This is optional and aims at
    providing default locations in which the program can be searched to avoid manual selection"""

    @staticmethod
    def itk_snap_check(rootdir=''):
        """checks for common folders in different platforms in which ITK-snap may be saved. """

        rootdir = ROOTDIR if not rootdir else rootdir
        platform = sys.platform
        cfg = Configuration.load_config(ROOTDIR)         # TODO: include the version stored in dependencies.py
        if not cfg['folders']['path2itksnap']:
            if platform == 'linux':
                default_folders = ["/etc/bin/", "/usr/lib/snap-3.6.0", "/usr/lib/snap-3.6.0/ITK-SNAP",
                                   os.path.join(rootdir, 'ext', 'snap-3.6.0')]
            elif platform == 'macos' or platform == 'darwin':
                default_folders = ['/Applications/ITK-SNAP.app/']

            try:
                folder = [folder_id for folder_id in default_folders if os.path.isfile(os.path.join(folder_id, "ITK-SNAP"))]
            except KeyError:
                folder = QFileDialog.getExistingDirectory('Please indicate location of ITK-SNAP.')
        else:
            folder = cfg['folders']['path2itksnap']
        # Here a dialog is needed in case folder has many flags to folders with itk-snap

        return folder[0]
