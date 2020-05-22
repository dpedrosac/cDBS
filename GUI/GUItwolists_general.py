#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import utils.HelperFunctions as HF
from utils.settingsDCM2NII import GuiSettingsDCM2NII
from utils import preprocDCM2NII
import private.allToolTips as setToolTips
import glob

from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QMainWindow, QListWidget


class TwoListGUI(QMainWindow):
    """ This is a GUI which aims at selecting the folders/subjects of which will be transformed usig one of the
    available options. It allows batch processing of data and in some cases changing the settings"""

    def __init__(self, option_gui='displayN4corr', _rootdir='', _title='', parent=None):
        super().__init__(parent)

        if not _title:
            title = 'Two list GUI for further processing data'

        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) if not _rootdir \
            else _rootdir

        self.setFixedSize(900, 600)
        self.setWindowTitle(title)
        self.table_widget = ContentTwoListGUI(rootdir, option_gui)
        self.setCentralWidget(self.table_widget)
        self.show()


class ContentTwoListGUI(QWidget):

    def __init__(self, rootdir, _option_gui, parent=None):
        super(QWidget, self).__init__(parent)
        self.cfg = HF.LittleHelpers.load_config(rootdir)
        self.option_gui = _option_gui

        if self.option_gui == 'dcm2niix':
            if os.path.isdir(self.cfg["folders"]["dicom"]):
                self.working_dir = self.cfg["folders"]["dicom"]
            else:
                self.working_dir = os.getcwd()

            folderbox_title = "Directory (DICOM-files)"
            str_labelDir = 'DICOM DIR: {}'.format(self.working_dir)
        elif self.option_gui == "displayN4corr":
            if os.path.isdir(self.cfg["folders"]["nifti"]):
                self.working_dir = self.cfg["folders"]["nifti"]
            else:
                self.working_dir = os.getcwd()

            folderbox_title = "Directory (nifti-files)"
            input_string = "all debiased and original nifti-files are loaded to the viewer, for visualising results"
            str_labelDir = 'nifti DIR: {}'.format(self.working_dir)
        else:
            HF.LittleHelpers.msg_box(text='Please provide a valid option such as "dcm2niix" or "displayN4corr". '
                                          'Terminating the GUI',
                                     title='Wrong input as option')
            ex.close()
            return

        # Create general layout
        self.tot_layout = QVBoxLayout(self)
        self.mid_layout = QHBoxLayout(self)

        # ============================    Create upper of  GUI, i.e. working directory   ============================
        self.folderboxDcm2nii = QGroupBox(folderbox_title)
        self.HBoxUpperDcm2nii = QVBoxLayout(self.folderboxDcm2nii)
        self.label_workingdir = QLabel(str_labelDir)
        self.HBoxUpperDcm2nii.addWidget(self.label_workingdir)

        self.btn_dicomdir = QPushButton('Change working \ndirectory')
        self.btn_dicomdir.setFixedSize(150, 40)
        self.btn_dicomdir.clicked.connect(self.change_workingdir)

        self.btn_savedir = QPushButton('Save directory \nto config file')
        self.btn_savedir.setFixedSize(150, 40)
        if self.option_gui != "dcm2niix":
            self.btn_savedir.setEnabled(False)

        self.btn_savedir.setToolTip(HF.LittleHelpers.split_lines(setToolTips.saveDirButton()))
        self.btn_savedir.clicked.connect(self.save_cfg_dicomdir)

        hlay_upper = QHBoxLayout()
        hlay_upper.addWidget(self.btn_dicomdir)
        hlay_upper.addWidget(self.btn_savedir)
        hlay_upper.addStretch(1)
        self.HBoxUpperDcm2nii.addLayout(hlay_upper)

        # ====================    Create Content for Lists, i.e. input/output      ====================
        self.listboxInputDcm2nii = QGroupBox('Available items in working directory')
        self.listboxInput = QVBoxLayout(self.listboxInputDcm2nii)
        self.mInput = QListWidget()
        self.listboxInput.addWidget(self.mInput)

        self.mButtonToAvailable = QPushButton("<<")
        self.mBtnMoveToAvailable = QPushButton(">")
        self.mBtnMoveToSelected = QPushButton("<")
        self.mButtonToSelected = QPushButton(">>")
        self.mBtnUp = QPushButton("Up")
        self.mBtnDown = QPushButton("Down")

        self.listboxOutputDcm2nii = QGroupBox('Items to process')
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
        if self.option_gui != "dcm2niix":
            self.btn_preferences.setEnabled(False)

        self.btn_run_command = QPushButton("Run Processing")
        if self.option_gui == "dcm2niix":
            self.btn_run_command.setToolTip(setToolTips.run_dcm2niix())
        else:
            self.btn_run_command.setToolTip(setToolTips.run_processing(input_string))
        self.btn_run_command.clicked.connect(self.start_process)

        hlay_bottom = QHBoxLayout()
        hlay_bottom.addStretch(1)
        hlay_bottom.addWidget(self.btn_preferences)
        hlay_bottom.addWidget(self.btn_run_command)
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
            items = self.create_folderlist(read_directory=self.working_dir)
            self.addAvailableItems(items)
        except FileExistsError:
            print('{} without any valid files/folders, continuing ...'.format(self.working_dir))

        self.update_buttons_status()
        self.connections()

    @staticmethod
    def create_folderlist(read_directory):
        """takes a folder and creates a list of all available subjects in this folder"""
        list_all = [name for name in os.listdir(read_directory)
                    if os.path.isdir(os.path.join(read_directory, name))]

        if list_all == '':
            list_subj = 'No available subjects, please make sure folders/items are present and correct prefix is given'
        else:
            list_subj = list_all
            list_subj = set(list_subj)

        return list_subj

    def addAvailableItems(self, items):
        """adds the available Items in the directory to read from into list; error message is dropped if 0 available"""

        if len(items) == 0:
            buttonReply = QMessageBox.question(self, 'No files/folders in the selected directory',
                                               'There are no subjects available in the current working directory ({}). '
                                               'Do you want to change to a different one?'.format(self.working_dir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_workingdir()
        else:
            self.mInput.addItems(items)

    def change_workingdir(self):
        """A new window appears in which the working directory can be set; besides, data is stored in the preferences
        file so that they will be loaded automatically next time"""

        self.working_dir = QFileDialog.getExistingDirectory(self, directory=self.working_dir,
                                                            caption='Change working directory')
        self.label_workingdir.setText('Working DIR: {}'.format(self.working_dir))
        self.mInput.clear()

        items = self.create_folderlist(read_directory=self.working_dir)
        self.addAvailableItems(items)
        if self.option_gui == "dcm2niix":
            self.cfg["folders"]["dicom"] = self.working_dir
            HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)

    def save_cfg_dicomdir(self):
        """Function intended to save the DICOM directory once button is pressed"""
        self.cfg["folders"]["dicom"] = self.working_dir
        HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)
        HF.LittleHelpers.msg_box(text="Folder changed in the configuration file to {}".format(self.working_dir),
                                 title='Changed folder')

    def settings_show(self):
        """Opens a new GUI in which the settings for the transformation con be changed and saved to config file"""
        self.settingsGUI = GuiSettingsDCM2NII()
        self.settingsGUI.show()

    def start_process(self):

        folderlist = []
        [folderlist.append(self.mOutput.item(x).text()) for x in range(self.mOutput.count())]
        print('in total, {} folders were selected'.format(len(folderlist)))

        if not folderlist:
            HF.LittleHelpers.msg_box(text="At least one folder with data must be selected!",
                                     title='No directory selected')
        elif len(folderlist) != 0 and self.option_gui == "dcm2niix":
            preprocDCM2NII.PreprocessDCM(folderlist)
        elif len(folderlist) != 0 and self.option_gui == "displayN4corr":
            input_folder = self.cfg["folders"]["nifti"]
            files_corrected = []
            [files_corrected.extend(glob.glob(os.path.join(input_folder, x + "/" +
                                                           self.cfg["preprocess"]["ANTsN4"]["prefix"] + "*")))
             for x in folderlist]
            files_original = [''.join(x.split(self.cfg["preprocess"]["ANTsN4"]["prefix"])) for x in files_corrected]

            viewer = 'itk-snap'  # to-date, only one viewer is available. May be changed in a future
            if len(files_corrected) + len(files_original) != 0:
                HF.LittleHelpers.load_imageviewer(viewer, files_original + files_corrected)

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
    ex = TwoListGUI()
    sys.exit(app.exec_())
