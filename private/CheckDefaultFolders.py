#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import QFileDialog


class FileLocation:
    """in this class, the default folders for the additional toolboxes can be stored. This is optional and aims at
    providing default locations in which the program can be searched to avoid manual selection"""

    @staticmethod
    def itk_snap_check(rootdir, platform=''):
        """checks for common folders in different platforms in which ITK-snap may be saved. """

        if not platform:
            import sys
            platform = sys.platform

        if platform == 'win32':
            default_folders = ['C:/Program Files/', 'C:/Program Files (x386)/',
                               os.path.join(rootdir, 'ext', 'snap-3.6.0')]
        elif platform == 'linux':
            default_folders = ["/etc/bin/", "/usr/lib/snap-3.6.0", "/usr/lib/snap-3.6.0/ITK-SNAP",
                               os.path.join(rootdir, 'ext', 'snap-3.6.0')]
        elif platform == 'macos':
            default_folders = ["/Applications/snap-3.6.0"]

        try:
            folder = [folder_id for folder_id in default_folders if os.path.isfile(os.path.join(folder_id, "ITK-SNAP"))]
        except KeyError:
            folder = QFileDialog.getExistingDirectory('Please indicate location of ITK-SNAP.')

        # Here a dialog is needed in case folder has many flags to folders with itk-snap

        return folder[0]
