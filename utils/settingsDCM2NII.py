#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import yaml
import os

from utils.HelperFunctions import Output, Configuration, FileOperations, MatlabEquivalent
import private.allToolTips as setToolTips
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, \
    QPushButton, QButtonGroup, QRadioButton, QLineEdit, QMessageBox
from dependencies import ROOTDIR, FILEDIR


class GuiSettingsDCM2NII(QWidget):
    """ This is a rather simple helper GUI, which aims at setting the options for Chris Rordens dcm2nii routines to
    convert DICOM files/folders to NIFTI"""

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent=None)
        self.cfg = Configuration.load_config(ROOTDIR)

        # General appearance of the GUI
        self.setFixedSize(600, 400)
        self.setWindowTitle('Settings for converting DICOM to NIFTI files')
        self.show()

        # Create general layout
        self.layout = QVBoxLayout(self)

        # ==============================    Create Content for First Box   ==============================
        self.optionbox = QGroupBox('Settings for dcm2niix')
        self.settings_list = QVBoxLayout(self.optionbox)

        self.labelBIDS = QLabel('Do you want to create a BIDS sidecar?\t\t')
        self.labelBIDS.setToolTip(setToolTips.LabelBIDS())
        self.btngroup_BIDS = QButtonGroup()
        self.rbtnBIDSy = QRadioButton('yes')
        self.rbtnBIDSn = QRadioButton('no')
        self.rbtnBIDSy.toggled.connect(self.onClickedRBTN_BIDS)
        self.rbtnBIDSn.toggled.connect(self.onClickedRBTN_BIDS)

        self.btngroup_BIDS.addButton(self.rbtnBIDSy)
        self.btngroup_BIDS.addButton(self.rbtnBIDSn)
        lay1 = QHBoxLayout()
        lay1.addWidget(self.labelBIDS)
        lay1.addWidget(self.rbtnBIDSy)
        lay1.addWidget(self.rbtnBIDSn)
        lay1.addStretch()

        self.labelVerbosity = QLabel('Please indicate the amount of \noutput from dcm2nii ("verbosity")\t\t\t')
        self.labelVerbosity.setToolTip(setToolTips.LabelVerbosity())
        self.btngroup_verbosity = QButtonGroup()
        self.rbtnVerbosity0 = QRadioButton('0')
        self.rbtnVerbosity1 = QRadioButton('1')
        self.rbtnVerbosity2 = QRadioButton('2')
        self.rbtnVerbosity0.toggled.connect(self.onClickedRBTN_verbose)
        self.rbtnVerbosity1.toggled.connect(self.onClickedRBTN_verbose)
        self.rbtnVerbosity2.toggled.connect(self.onClickedRBTN_verbose)

        self.btngroup_verbosity.addButton(self.rbtnVerbosity0)
        self.btngroup_verbosity.addButton(self.rbtnVerbosity1)
        self.btngroup_verbosity.addButton(self.rbtnVerbosity2)

        lay2 = QHBoxLayout()
        lay2.addWidget(self.labelVerbosity)
        lay2.addWidget(self.rbtnVerbosity0)
        lay2.addWidget(self.rbtnVerbosity1)
        lay2.addWidget(self.rbtnVerbosity2)
        lay2.addStretch()

        self.labelCompression = QLabel('Do you want dcm2niix to compress the \nresulting NIFTI files?\t\t\t\t')
        self.labelCompression.setToolTip(setToolTips.CompressionDCM2NII())
        self.btngroup_Compression = QButtonGroup()
        self.rbtnCompressiony = QRadioButton('yes')
        self.rbtnCompressionn = QRadioButton('no')
        self.rbtnCompressiony.toggled.connect(self.onClickedRBTN_Compression)
        self.rbtnCompressionn.toggled.connect(self.onClickedRBTN_Compression)

        self.btngroup_Compression.addButton(self.rbtnCompressiony)
        self.btngroup_Compression.addButton(self.rbtnCompressionn)
        lay3 = QHBoxLayout()
        lay3.addWidget(self.labelCompression)
        lay3.addWidget(self.rbtnCompressiony)
        lay3.addWidget(self.rbtnCompressionn)
        lay3.addStretch()

        self.labelFilename = QLabel('What should output filenames be like?\t\t')
        self.labelFilename.setToolTip(setToolTips.LabelFilenameDCM2NII())
        self.lineEditFilename = QLineEdit()
        regex = QtCore.QRegExp("[a-z-A-Z-0-9%_]+")
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditFilename.setValidator(validator1)

        lay4 = QHBoxLayout()
        lay4.addWidget(self.labelFilename)
        lay4.addWidget(self.lineEditFilename)
        lay4.addStretch()

        self.labelIncludeFiles = QLabel('Please indicate the sequences to keep?\t\t')
        self.labelIncludeFiles.setToolTip(setToolTips.includeFilesDCM2NII())
        self.lineEditIncludeFiles = QLineEdit()
        regex = QtCore.QRegExp("[a-z-A-Z-0-9,_]+")
        validator2 = QtGui.QRegExpValidator(regex)
        self.lineEditIncludeFiles.setValidator(validator2)

        lay5 = QHBoxLayout()
        lay5.addWidget(self.labelIncludeFiles)
        lay5.addWidget(self.lineEditIncludeFiles)
        lay5.addStretch()

        self.labelReorientCrop = QLabel('Do you want dcm2niix to reorient \nand crop resulting NIFTI-files?\t\t\t')
        self.labelReorientCrop.setToolTip(setToolTips.LabelReorientCrop())
        self.btngroup_ReorientCrop = QButtonGroup()
        self.rbtnReorientCropy = QRadioButton('yes')
        self.rbtnReorientCropn = QRadioButton('no')
        self.rbtnReorientCropy.toggled.connect(self.onClickedRBTN_ReorientCrop)
        self.rbtnReorientCropn.toggled.connect(self.onClickedRBTN_ReorientCrop)

        self.btngroup_ReorientCrop.addButton(self.rbtnReorientCropy)
        self.btngroup_ReorientCrop.addButton(self.rbtnReorientCropn)
        lay6 = QHBoxLayout()
        lay6.addWidget(self.labelReorientCrop)
        lay6.addWidget(self.rbtnReorientCropy)
        lay6.addWidget(self.rbtnReorientCropn)
        lay6.addStretch()

        self.settings_list.addLayout(lay1)
        self.settings_list.addLayout(lay2)
        self.settings_list.addLayout(lay3)
        self.settings_list.addLayout(lay4)
        self.settings_list.addLayout(lay5)
        self.settings_list.addLayout(lay6)

        # ====================    Create Content for Buttons at the Bottom      ====================
        layout_bottom = QHBoxLayout()
        self.buttonsave = QPushButton('Save settings \nand return')
        self.buttonsave.clicked.connect(self.close)
        self.buttondefault = QPushButton('Load Default \nsettings')
        self.buttondefault.clicked.connect(self.load_default_DCM2NIIsettings)

        layout_bottom.addStretch(1)
        layout_bottom.addWidget(self.buttonsave)
        layout_bottom.addWidget(self.buttondefault)

        # ====================    Set Content of box and buttoms to General Layout     =======================
        self.layout.addWidget(self.optionbox)
        self.layout.addLayout(layout_bottom)
        self.get_settings_from_config()

    def get_settings_from_config(self):
        """function which enters the settings according to cfg variable which is loaded"""

        if self.cfg == "":
            print()
            Output.msg_box(title="Warning", text="No default settings found, please double check the folder content. "
                                          "Continuing with same settings.")
        else:
            if self.cfg["preprocess"]["dcm2nii"]["BIDSsidecar"] == 'yes':
                self.rbtnBIDSy.setChecked(True)
            else:
                self.rbtnBIDSn.setChecked(True)

            if self.cfg["preprocess"]["dcm2nii"]["OutputCompression"] == 'yes':
                self.rbtnCompressiony.setChecked(True)
            else:
                self.rbtnCompressionn.setChecked(True)

            if self.cfg["preprocess"]["dcm2nii"]["Verbosity"] == 0:
                self.rbtnVerbosity0.setChecked(True)
            elif self.cfg["preprocess"]["dcm2nii"]["Verbosity"] == 1:
                self.rbtnVerbosity1.setChecked(True)
            else:
                self.rbtnVerbosity2.setChecked(True)

            self.lineEditFilename.setText(self.cfg["preprocess"]["dcm2nii"]["OutputFileStruct"])
            self.lineEditIncludeFiles.setText(self.cfg["preprocess"]["dcm2nii"]["IncludeFiles"])

            if self.cfg["preprocess"]["dcm2nii"]["ReorientCrop"] == 'yes':
                self.rbtnReorientCropy.setChecked(True)
            else:
                self.rbtnReorientCropn.setChecked(True)

    def closeEvent(self, event):
        """saves the settings found here as a yaml file which may be loaded the next time as the configuration used"""

        self.cfg["preprocess"]["dcm2nii"]["OutputFileStruct"] = self.lineEditFilename.text()
        self.cfg["preprocess"]["dcm2nii"]["IncludeFiles"] = self.lineEditIncludeFiles.text()
        Configuration.save_config(self.cfg["folders"]["rootdir"], self.cfg)
        event.accept()

    def load_default_DCM2NIIsettings(self):
        """loads the default settings as per the file in the private folder; for that a confirmation is necessary"""

        ret = QMessageBox.question(self, 'MessageBox', "Do you really want to restore default settings for dcm2niix?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            with open(os.path.join(ROOTDIR, 'private/') + 'config_imagingTBdef.yaml', 'r') as cfg:
                cfg_temp = yaml.safe_load(cfg)
                self.cfg["preprocess"]["dcm2nii"] = cfg_temp["preprocess"]["dcm2nii"]
        self.get_settings_from_config()

    # In the next few lines, actions are taken when rbtns are pressed; Principally, button is checked and cfg updated
    @QtCore.pyqtSlot()
    def onClickedRBTN_BIDS(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["dcm2nii"]["BIDSsidecar"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_verbose(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["dcm2nii"]["Verbosity"] = int(self.sender().text())

    @QtCore.pyqtSlot()
    def onClickedRBTN_Compression(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["dcm2nii"]["OutputCompression"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_ReorientCrop(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["dcm2nii"]["ReorientCrop"] = self.sender().text()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GuiSettingsDCM2NII()
    sys.exit(app.exec_())
