#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QPushButton, QMainWindow, QListWidget

import private.allToolTips as setToolTips
from dependencies import ROOTDIR
from utils import preprocDCM2NII
from utils.HelperFunctions import Configuration, FileOperations, Output
from utils.settingsDCM2NII import GuiSettingsDCM2NII


class MainGuiDcm2nii(QMainWindow):
    """ GUI aiming at selecting folders/subjects of whom imaging will be transformed from DICOM to NIFTI-files using
    Chris Rordens dcm2niix routines. Allows batch processing and changing some settings to later run wrapper """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(900, 600)
        self.setWindowTitle('Batch convert DICOM files to NIFTI using dcm2niix ')
        self.table_widget = ContentGuiDCM2Nii(self)
        self.setCentralWidget(self.table_widget)
        self.show()


class ContentGuiDCM2Nii(QWidget):
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        # Load configuration files and general settings
        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['dicom']):
            self.dicomdir = self.cfg['folders']['dicom']
        else:
            self.dicomdir = FileOperations.set_wdir_in_config(self.cfg, foldername='dicom', init=True)

        self.cfg['folders']['dicom'] = self.dicomdir
        self.cfg['folders']['rootdir'] = ROOTDIR
        Configuration.save_config(ROOTDIR, self.cfg)

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
        self.btn_savedir.setToolTip(Output.split_lines(setToolTips.saveDirButton()))
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
        self.btn_run_dcm2niix.setToolTip(setToolTips.run_dcm2niix())
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
            items = FileOperations.list_folders(inputdir=self.cfg['folders']['dicom'], prefix='', files2lookfor='')
            items = self.check_for_complete_input(list(items))
            self.add_available_subj(items)
        except FileExistsError:
            print('{} without any valid files/folders, continuing ...'.format(self.dicomdir))

        self.update_buttons_status()
        self.connections()

    def check_for_complete_input(self, item_list, modalities = ['CT', 'MRI']):
        """this function ensures, that only those subjects are displayed where MRI and CT data is available in the
        correct folders (surname_nameMRI or surname_nameCT)"""

        available_subjects = set([re.split(r'CT|MRI', x)[0] for x in list(item_list)])
        item_list_complete = []
        [item_list_complete.append(subj) for subj in list(available_subjects) if
         all([os.path.isdir(os.path.join(self.dicomdir, subj + y)) for y in modalities])]

        if len(available_subjects) != len(item_list_complete):
            incomplete = list(set(available_subjects) - set(item_list_complete))
            Output.msg_box(text="There is incomplete data or directories have unknown names. Please ensure the presence"
                                " of two folders (surname_nameCT) and (surname_nameMRI) for:"
                                "\n{}".format(''.join(' -> {}\n'.format(c) for c in incomplete)),
                           title="Incomplete data")
        return set(item_list_complete)

    def add_available_subj(self, items):
        """adds available subjects in the cwd to item list; PyQT5 dialog is opened if none available"""

        if len(items) == 0:
            buttonReply = QMessageBox.question(self, 'No files in the DICOM directory',
                                               'No subjects available in the CWD: {}. '
                                               'Do you want to change to different one?'.format(self.dicomdir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_dicomdir()
        else:
            self.mInput.addItems(items)

    def change_dicomdir(self):
        """A new window appears in which the working directory can be set; besides, data is stored in the preferences
        file so that they will be loaded automatically next time"""

        self.dicomdir = FileOperations.set_wdir_in_config(self.cfg, foldername='dicom')
        self.label_dicomdir.setText('dicomDIR: {}'.format(self.dicomdir))
        self.cfg['folders']['dicom'] = self.dicomdir

        self.mInput.clear()
        items = FileOperations.list_folders(inputdir=self.cfg['folders']['dicom'], prefix='', files2lookfor='')
        self.add_available_subj(items)

    def save_cfg(self):
        """Function intended to save the DICOM directory once button is pressed"""

        self.cfg['folders']['dicom'] = self.dicomdir
        Configuration.save_config(self.cfg['folders']['rootdir'], self.cfg)
        Output.msg_box(text="Folder changed in configuration to {}".format(self.dicomdir), title="Changed folder")

    def settings_show(self):
        """Opens a new GUI in which the settings for the tansformation con be changed and saved to config file"""
        self.settingsGUI = GuiSettingsDCM2NII()
        self.settingsGUI.show()

    def start_converting(self):
        folderlist = []
        [folderlist.append(self.mOutput.item(x).text()) for x in range(self.mOutput.count())]
        print('in total, {} folders were selected'.format(len(folderlist)))

        if not folderlist:
            Output.msg_box(text="At least one folder with DICOM data needed!", title="No directory selected")
        else:
            preprocDCM2NII.PreprocessDCM(folderlist)

    # ====================    Actions when buttons are pressed      ====================
    @QtCore.pyqtSlot()
    def update_buttons_status(self):
        self.mBtnUp.setDisabled(not bool(self.mOutput.selectedItems()) or self.mOutput.currentRow() == 0)
        self.mBtnDown.setDisabled(not bool(self.mOutput.selectedItems()) or
                                  self.mOutput.currentRow() == (self.mOutput.count() - 1))
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
