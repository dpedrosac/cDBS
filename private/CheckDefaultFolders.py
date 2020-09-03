#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import QFileDialog
from dependencies import ROOTDIR

class FileLocation:
    """in this class, the default folders for the additional toolboxes can be stored. This is optional and aims at
    providing default locations in which the program can be searched to avoid manual selection"""

    @staticmethod
    def itk_snap_check(rootdir, platform=''):
        """checks for common folders in different platforms in which ITK-snap may be saved. """

        # TODO: include the version stored in dependencies.py and move this part to HelperFunctions.py and Imaging class
        if not platform:
            import sys
            platform = sys.platform

        if platform == 'linux':
            default_folders = ["/etc/bin/", "/usr/lib/snap-3.6.0", "/usr/lib/snap-3.6.0/ITK-SNAP",
                               os.path.join(rootdir, 'ext', 'snap-3.6.0')]
        elif platform == 'macos' or platform == 'darwin':
            default_folders = ["/Applications/ITK-SNAP.app/"]

        try:
            folder = [folder_id for folder_id in default_folders if os.path.isfile(os.path.join(folder_id, "ITK-SNAP"))]
        except KeyError:
            folder = QFileDialog.getExistingDirectory('Please indicate location of ITK-SNAP.')

        # Here a dialog is needed in case folder has many flags to folders with itk-snap

        return folder[0]
