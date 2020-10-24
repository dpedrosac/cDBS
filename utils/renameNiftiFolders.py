#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QPushButton, QLineEdit

import private.allToolTips as setToolTips
from dependencies import ROOTDIR
from utils.HelperFunctions import Configuration, FileOperations, Output


class RenameFolderNames(QWidget):
    """ GUI providing necessary information to change directory names in the NIFTI folder e.g. from subjxx to DBSxx"""

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.setFixedSize(600, 250)
        self.setWindowTitle("Batch convert subjects' folder names")
        self.show()

        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['nifti']):
            self.niftidir = self.cfg['folders']['nifti']
        else:
            self.niftidir = FileOperations.set_wdir_in_config(self.cfg, foldername='nifti', init=True)
        Configuration.save_config(ROOTDIR, self.cfg)

        # Create general layout
        self.layout = QVBoxLayout(self)
        self.BottomButtons = QHBoxLayout()
        self.GroupBox1 = QGroupBox("Change prefix of folders:")
        self.EnterDataLayout = QHBoxLayout(self.GroupBox1)

        self.lblPrefix = QLabel('Old suffix: {}'.format('\u0332'.join(self.cfg['folders']['prefix'])))
        self.Arrow = QLabel('->\t')
        self.lineEditChangedPrefix = QLineEdit()
        self.lineEditChangedPrefix.setPlaceholderText("Enter here the new folder prefix (e.g. 'IPS-patient')")
        regex = QtCore.QRegExp("[a-z-A-Z-0-9_.]+")
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditChangedPrefix.setValidator(validator1)
        self.lineEditChangedPrefix.setToolTip(setToolTips.renameFoldersInput())

        self.EnterDataLayout.addWidget(self.lblPrefix)
        self.EnterDataLayout.addWidget(self.Arrow)
        self.EnterDataLayout.addWidget(self.lineEditChangedPrefix)

        self.btnOK = QPushButton('OK')
        self.btnOK.clicked.connect(self.on_OKBtn_clicked)
        self.btnCancel = QPushButton('Cancel')
        self.btnCancel.clicked.connect(self.close)
        self.BottomButtons.addStretch()
        self.BottomButtons.addWidget(self.btnOK)
        self.BottomButtons.addWidget(self.btnCancel)

        self.layout.addWidget(self.GroupBox1)
        self.layout.addLayout(self.BottomButtons)

    # ====================    Actions when buttons are pressed      ====================
    @QtCore.pyqtSlot()
    def on_OKBtn_clicked(self):
        prefix = self.cfg['folders']['prefix']
        new_prefix = self.lineEditChangedPrefix.text()
        if not new_prefix:
            Output.msgBox(text="No prefix was entered!", title="Warning")
        elif new_prefix.endswith('_') or new_prefix.endswith('.'):
            Output.msgBox(text="Special characters '_' or '.' should be avoided at the end", title="Warning")
        else:
            msg = 'Are you sure you want to change the prefix of all folders in the Nifti-Directory ' \
                  'from \n\n{:>10} \tto {:>10}?'.format(prefix, new_prefix)

            ret = QMessageBox.question(self, 'MessageBox', msg,
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.cfg['folders']['prefix'] = self.lineEditChangedPrefix.text()
                self.ChangeFolderNames(prefix, new_prefix)
                Configuration.save_config(ROOTDIR, self.cfg)
                self.close()

    def ChangeFolderNames(self, prefix, new_prefix):
        """Changes the names of the folders in the NIFTI directory according to the settings"""

        list_pre = [name for name in os.listdir(self.niftidir)
                    if (os.path.isdir(os.path.join(self.niftidir, name)) and prefix in name)]
        temp = re.compile('([a-zA-Z._-]+)([0-9]+)')
        list_post = [new_prefix + temp.match(numbers).groups()[1] for numbers in list_pre]
        [os.rename(os.path.join(self.niftidir, prename), os.path.join(self.niftidir, postname))
            for prename, postname in zip(list_pre, list_post)]

    def closeEvent(self, event):
        """saves the settings found here as a yaml file and closes the GUI"""
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = RenameFolderNames()
    sys.exit(app.exec_())
