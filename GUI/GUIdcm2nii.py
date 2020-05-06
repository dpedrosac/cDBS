#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import utils.HelperFunctions as HF
from utils.settingsDCM2NII import GuiSettingsDCM2NII
from utils import preprocDCM2NII
import private.allToolTips as setToolTips

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
     QFileDialog, QPushButton, QMainWindow, QListWidget


class MainGuiDcm2nii(QMainWindow):
    """ This is a GUI which aims at selecting the folders/subjects of whom imaging will be transformed from DICOM to
    NIFTI files using Chris Rordens dcm2niix routines. It allows batch processing of data and changing some of the settings
    to later run a wrapper for the code"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(900, 600)
        self.setWindowTitle('Batch convert DICOM files to NIFTI using dcm2niix ')
        self.table_widget = ContentGuiDcm2nii(self)
        self.setCentralWidget(self.table_widget)
        self.show()


class ContentGuiDcm2nii(QWidget):

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        # Load configuration files and general settings
        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.cfg = HF.LittleHelpers.load_config(rootdir)
        if os.path.isdir(self.cfg["folders"]["dicom"]):
            self.dicomdir = self.cfg["folders"]["dicom"]
        else:
            self.dicomdir = os.getcwd()

        # Create general layout
        self.tot_layout = QVBoxLayout(self)
        self.mid_layout = QHBoxLayout(self)

        # ============================    Create upper of  GUI, i.e. working directory   ============================
        self.folderboxDcm2nii = QGroupBox("Directory (DICOM-files)")
        self.HBoxUpperDcm2nii = QVBoxLayout(self.folderboxDcm2nii)
        self.label_dicomdir = QLabel('dicom DIR: {}'.format(self.dicomdir))
        self.HBoxUpperDcm2nii.addWidget(self.label_dicomdir)

        self.btn_dicomdir = QPushButton('Change working \ndirectory')
        self.btn_dicomdir.setFixedSize(150, 40)
        self.btn_dicomdir.clicked.connect(self.change_dicomdir)
        self.btn_savedir = QPushButton('Save directory \nto config file')
        self.btn_savedir.setFixedSize(150, 40)
        self.btn_savedir.setToolTip(HF.LittleHelpers.split_lines(setToolTips.saveDirButton()))
        self.btn_savedir.clicked.connect(self.save_cfg)

        hlay_upper = QHBoxLayout()
        hlay_upper.addWidget(self.btn_dicomdir)
        hlay_upper.addWidget(self.btn_savedir)
        hlay_upper.addStretch(1)
        self.HBoxUpperDcm2nii.addLayout(hlay_upper)

        # ====================    Create Content for Lists, i.e. input/output      ====================
        self.listboxInputDcm2nii = QGroupBox('Available subjects in working directory')
        self.listboxInput = QVBoxLayout(self.listboxInputDcm2nii)
        self.mInput = QListWidget()
        self.listboxInput.addWidget(self.mInput)

        self.mButtonToAvailable = QPushButton("<<")
        self.mBtnMoveToAvailable = QPushButton(">")
        self.mBtnMoveToSelected = QPushButton("<")
        self.mButtonToSelected = QPushButton(">>")
        self.mBtnUp = QPushButton("Up")
        self.mBtnDown = QPushButton("Down")

        self.listboxOutputDcm2nii = QGroupBox('Subjects to process')
        self.listboxOutput = QVBoxLayout(self.listboxOutputDcm2nii)
        self.mOutput = QListWidget()
        self.listboxOutput.addWidget(self.mOutput)

        # First column on the left side
        vlay = QVBoxLayout()
        vlay.addStretch()
        vlay.addWidget(self.mBtnMoveToAvailable)
        vlay.addWidget(self.mBtnMoveToSelected)
        vlay.addStretch()
        vlay.addWidget(self.mButtonToAvailable)
        vlay.addWidget(self.mButtonToSelected)
        vlay.addStretch()

        # Second column on the right side
        vlay2 = QVBoxLayout()
        vlay2.addStretch()
        vlay2.addWidget(self.mBtnUp)
        vlay2.addWidget(self.mBtnDown)
        vlay2.addStretch()

        # ====================    Lower part of GUI, i.e. Preferences/Start estimation      ====================
        self.btn_preferences = QPushButton("Preferences")
        self.btn_preferences.clicked.connect(self.settings_show)
        self.btn_run_dcm2niix = QPushButton("Run dcm2niix")
        self.btn_run_dcm2niix.clicked.connect(self.start_converting)

        hlay_bottom = QHBoxLayout()
        hlay_bottom.addStretch(1)
        hlay_bottom.addWidget(self.btn_preferences)
        hlay_bottom.addWidget(self.btn_run_dcm2niix)
        hlay_bottom.addStretch()

        # ====================    Set all contents to general Layout     =======================
        self.mid_layout.addWidget(self.listboxInputDcm2nii)
        self.mid_layout.addLayout(vlay)
        self.mid_layout.addWidget(self.listboxOutputDcm2nii)
        self.mid_layout.addLayout(vlay2)

        self.tot_layout.addWidget(self.folderboxDcm2nii)
        self.tot_layout.addLayout(self.mid_layout)
        self.tot_layout.addLayout(hlay_bottom)

        try:
            self.mInput.clear()
            items = self.read_subjlist(dicomdir=self.cfg["folders"]["dicom"])
            self.addAvailableSubj(items)
        except FileExistsError:
            print('{} without any valid files/folders, continuing ...'.format(self.dicomdir))

        self.update_buttons_status()
        self.connections()

    @staticmethod # TODO similarly to GUIToolbox, this is so pivotal that it MUST be contained in helper functions
    def read_subjlist(dicomdir):
        """takes a folder and creates a list of all available subjects in this folder"""
        list_all = [name for name in os.listdir(dicomdir)
                    if os.path.isdir(os.path.join(dicomdir, name))]

        if list_all == '':
            list_subj = 'No available subjects, please make sure DICOM folders are present and correct prefix is given'
        else:
            list_subj = list_all
            list_subj = set(list_subj)

        return list_subj

    def addAvailableSubj(self, items):
        """adds the available subjects in the working directory into the items list;
        an error message is dropped if none available"""

        if len(items) == 0:
            buttonReply = QMessageBox.question(self, 'No files in the dicomdir', 'There are no subjects available '
                                               'in the current working directory ({}). Do you want to '
                                               ' change to a different one?'.format(self.dicomdir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_dicomdir()
        else:
            self.mInput.addItems(items)

    def change_dicomdir(self):
        """A new window appears in which the working directory can be set; besides, data is stored in the preferences
        file so that they will be loaded automatically next time"""

        self.dicomdir = QFileDialog.getExistingDirectory(self, 'title')
        self.label_dicomdir.setText('dicomDIR: {}'.format(self.dicomdir))
        self.mInput.clear()

        items = self.read_subjlist(dicomdir=self.cfg["folders"]["dicom"])
        self.addAvailableSubj(items)
        self.cfg["folders"]["dicom"] = self.dicomdir
        HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)

    def save_cfg(self):
        """Function intended to save the DICOM directory once button is pressed"""
        self.cfg["folders"]["dicom"] = self.dicomdir
        HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)
        HF.LittleHelpers.msg_box(text="Folder changed in the configuration file to {}".format(self.dicomdir),
                                 title='Changed folder')

    def settings_show(self):
        """Opens a new GUI in which the settings for the tansformation con be changed and saved to config file"""
        self.settingsGUI = GuiSettingsDCM2NII()
        self.settingsGUI.show()

    def start_converting(self):
        folderlist = []
        [folderlist.append(self.mOutput.item(x).text()) for x in range(self.mOutput.count())]
        print('in total, {} folders were selected'.format(len(folderlist)))

        if not folderlist:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("At least one folder with DICOM data must be selected!")
            msg.setWindowTitle("No directory selected")
            msg.exec_()
        else:
            preprocDCM2NII.PreprocessDCM(folderlist)

    @QtCore.pyqtSlot()
    def update_buttons_status(self):
        self.mBtnUp.setDisabled(not bool(self.mOutput.selectedItems()) or self.mOutput.currentRow() == 0)
        self.mBtnDown.setDisabled(not bool(self.mOutput.selectedItems()) or self.mOutput.currentRow() == (self.mOutput.count() - 1))
        self.mBtnMoveToAvailable.setDisabled(not bool(self.mInput.selectedItems()) or self.mOutput.currentRow() == 0)
        self.mBtnMoveToSelected.setDisabled(not bool(self.mOutput.selectedItems()))
        self.mButtonToAvailable.setDisabled(not bool(self.mOutput.selectedItems()))
        self.mButtonToSelected.setDisabled(not bool(self.mInput.selectedItems()))

    def connections(self):
        self.mInput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mOutput.itemSelectionChanged.connect(self.update_buttons_status)
        self.mBtnMoveToAvailable.clicked.connect(self.on_mBtnMoveToAvailable_clicked)
        self.mBtnMoveToSelected.clicked.connect(self.on_mBtnMoveToSelected_clicked)
        self.mButtonToAvailable.clicked.connect(self.on_mButtonToAvailable_clicked)
        self.mButtonToSelected.clicked.connect(self.on_mButtonToSelected_clicked)
        self.mBtnUp.clicked.connect(self.on_mBtnUp_clicked)
        self.mBtnDown.clicked.connect(self.on_mBtnDown_clicked)

    @QtCore.pyqtSlot()
    def on_mBtnMoveToAvailable_clicked(self):
        self.mOutput.addItem(self.mInput.takeItem(self.mInput.currentRow()))

    @QtCore.pyqtSlot()
    def on_mBtnMoveToSelected_clicked(self):
        self.mInput.addItem(self.mOutput.takeItem(self.mOutput.currentRow()))

    @QtCore.pyqtSlot()
    def on_mButtonToAvailable_clicked(self):
        while self.mOutput.count() > 0:
            self.mInput.addItem(self.mOutput.takeItem(0))

    @QtCore.pyqtSlot()
    def on_mButtonToSelected_clicked(self):
        while self.mInput.count() > 0:
            self.mOutput.addItem(self.mInput.takeItem(0))

    @QtCore.pyqtSlot()
    def on_mBtnUp_clicked(self):
        row = self.mOutput.currentRow()
        currentItem = self.mOutput.takeItem(row)
        self.mOutput.insertItem(row - 1, currentItem)
        self.mOutput.setCurrentRow(row - 1)

    @QtCore.pyqtSlot()
    def on_mBtnDown_clicked(self):
        row = self.mOutput.currentRow()
        currentItem = self.mOutput.takeItem(row)
        self.mOutput.insertItem(row + 1, currentItem)
        self.mOutput.setCurrentRow(row + 1)

    def addSelectedItems(self, items):
        self.mOutput.addItems(items)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainGuiDcm2nii()
    sys.exit(app.exec_())
