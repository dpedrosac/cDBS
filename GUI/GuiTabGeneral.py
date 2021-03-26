#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QListWidget, QAbstractItemView

from utils.HelperFunctions import Output, Configuration, FileOperations
from GUI.GUIdcm2nii import MainGuiDcm2nii
from utils.renameNiftiFolders import RenameFolderNames
from GUI.GuiTwoLists_generic import TwoListGUI
from dependencies import ROOTDIR
import private.allToolTips as setToolTips


class GuiTabGeneral(QWidget):
    """General tab which enables import of DICOM files but also a set of distinct options such as viewing the
    metadata or displaying images in an external viewer and renaming folders"""

    def __init__(self, parent=None):
        super(GuiTabGeneral, self).__init__(parent)
        self.selected_subj_Gen = ''

        # General settings/variables/helper files needed needed at some point
        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['nifti']):
            self.niftidir = self.cfg['folders']['nifti']
        else:
            self.niftidir = FileOperations.set_wdir_in_config(self.cfg, foldername='nifti', init=True)
        # self.cfg['folders']['rootdir'] = ROOTDIR  ## changed 2021/02/23
        Configuration.save_config(ROOTDIR, self.cfg)

        self.lay = QHBoxLayout(self)
        self.tab = QWidget()

        # Customize tab
        # ==============================    Tab 1 - General   ==============================
        self.tab.layout = QHBoxLayout()
        self.tab.setLayout(self.tab.layout)
        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab = QGroupBox("Directory (NIFTI-files)")
        self.HBoxUpperLeftTab = QVBoxLayout(self.FolderboxTab)

        self.lblWdirTab = QLabel('wDIR: {}'.format(self.niftidir))
        self.HBoxUpperLeftTab.addWidget(self.lblWdirTab)

        self.btnChangeWdir = QPushButton('Change working directory')
        self.btnChangeWdir.setToolTip(setToolTips.ChangeWdirDICOM())
        self.btnChangeWdir.clicked.connect(self.change_wdir)

        self.btnReloadFilesTab = QPushButton('Reload files')
        self.btnReloadFilesTab.clicked.connect(self.run_reload_files)

        self.HBoxUpperLeftTab.addWidget(self.btnChangeWdir)
        self.HBoxUpperLeftTab.addWidget(self.btnReloadFilesTab)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.ActionsTab = QGroupBox("Functions")
        self.HBoxLowerLeftTab = QVBoxLayout(self.ActionsTab)
        self.btn_subj_details = QPushButton('Subject details')
        self.btn_subj_details.setToolTip(setToolTips.subjectDetails())
        self.btn_subj_details.clicked.connect(self.openDetails)

        self.btn_dcm2nii = QPushButton('Dcm2niix')
        self.btn_dcm2nii.setToolTip(setToolTips.runDCM2NII())
        self.btn_dcm2nii.clicked.connect(self.run_DCM2NII)

        self.btn_viewer = QPushButton('View available \nNIFTI-files in viewer')
        self.btn_viewer.setToolTip(setToolTips.displayFolderContent())
        self.btn_viewer.clicked.connect(self.show_nifti_files)

        self.btn_renaming = QPushButton('Rename\nfolders')
        self.btn_renaming.setToolTip(setToolTips.renameFolders())
        self.btn_renaming.clicked.connect(self.run_rename_folders)

        self.HBoxLowerLeftTab.addWidget(self.btn_subj_details)
        self.HBoxLowerLeftTab.addWidget(self.btn_dcm2nii)
        self.HBoxLowerLeftTab.addWidget(self.btn_viewer)
        self.HBoxLowerLeftTab.addWidget(self.btn_renaming)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox = QGroupBox('Available subjects')
        self.HBoxUpperRightTab = QVBoxLayout(self.listbox)
        self.availableNiftiTab = QListWidget()
        self.availableNiftiTab.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.availableNiftiTab.itemSelectionChanged.connect(self.change_list_item)

        itemsTab = FileOperations.list_folders(self.niftidir, prefix=self.cfg['folders']['prefix'])
        self.add_available_items(self.availableNiftiTab, itemsTab)
        self.HBoxUpperRightTab.addWidget(self.availableNiftiTab)

        # Combine all Boxes for General Tab Layout
        self.LeftboxTab = QGroupBox()
        self.HBoxTabLeft = QVBoxLayout(self.LeftboxTab)
        self.HBoxTabLeft.addWidget(self.FolderboxTab)
        self.HBoxTabLeft.addStretch()
        self.HBoxTabLeft.addWidget(self.ActionsTab)

        self.tab.layout.addWidget(self.LeftboxTab)
        self.tab.layout.addWidget(self.listbox)

        self.lay.addWidget(self.tab)

    # ------------------------- Start with the functions for lists and buttons in this tab  ------------------------- #
    def change_wdir(self):
        """A new window appears in which the working directory for NIFTI-files can be set; if set, this is stored
         in the configuration file, so that upon the next start there is the same folder selected automatically"""

        self.niftidir = FileOperations.set_wdir_in_config(self.cfg, foldername='nifti')
        self.lblWdirTab.setText('wDIR: {}'.format(self.niftidir))
        self.cfg['folders']['nifti'] = self.niftidir

        self.availableNiftiTab.clear()
        itemsChanged = FileOperations.list_folders(self.niftidir, self.cfg['folders']['prefix'])
        self.add_available_items(self.availableNiftiTab, itemsChanged)

    def run_reload_files(self):
        """Reloads files, e.g. after renaming them"""

        self.cfg = Configuration.load_config(ROOTDIR)
        self.availableNiftiTab.clear()
        itemsChanged = FileOperations.list_folders(self.cfg['folders']['nifti'], prefix=self.cfg['folders']['prefix'])
        self.add_available_items(self.availableNiftiTab, itemsChanged)

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""

        if self.sender() == self.availableNiftiTab:
            items = self.availableNiftiTab.selectedItems()
            self.selected_subj_Gen = []

            for i in range(len(items)):
                self.selected_subj_Gen.append(str(self.availableNiftiTab.selectedItems()[i].text()))

    def add_available_items(self, sending_list, items, msg="yes"):
        """adds the available subjects in the working directory into the items list;
        an error message is dropped if none available"""
        if len(items) == 0 and msg == "yes":
            buttonReply = QMessageBox.question(self, "No files in dir", "There are no subjects available in the "
                                                                        "working directory ({}). Change to different "
                                                                        "one?".format(self.niftidir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_wdir()
        else:
            items = list(items)
            items.sort(key=lambda folder_name: int(folder_name.split(self.cfg['folders']['prefix'])[1]))
            sending_list.addItems(items)

    def openDetails(self):
        """opens details file which has additional information on subjects """
        import subprocess

        fileName = os.path.join(self.cfg['folders']['nifti'], 'subjdetails.csv')
        if os.path.isfile(fileName):
            if sys.platform == 'linux':
                subprocess.Popen(['xdg-open', ''.join(fileName)])
            else:
                os.system('open"%s"' % fileName)
                subprocess.run(['open', fileName], check=True)
        else:
            Output.msg_box(text="Subject details unavailable, creating empty csv-file!", title="Detail file not found")
            subprocess.call(['touch', '{}subjdetails.csv'.format(self.cfg['folders']['nifti'])])

    def run_rename_folders(self):
        """Renames all folders with a similar prefix; After that manual reloading is necessary"""

        self.convertFolders = RenameFolderNames()
        self.convertFolders.show()

    def show_nifti_files(self):
        """this function opens a list dialog and enables selecting NIFTI files for e.g. check the content (identical
        function as in GUITabPreprocessANTs.py."""

        if not self.selected_subj_Gen:
            Output.msg_box(text="No folder selected. To proceed, please indicate what folder to process.",
                           title="No subject selected")
            return
        elif len(self.selected_subj_Gen) > 1:
            Output.msg_box(text="Please select only one folder to avoid excessive image load",
                           title="Number of selected files")
            return
        else:
            image_folder = os.path.join(self.cfg['folders']['nifti'], self.selected_subj_Gen[0])

        self.SelectFiles = TwoListGUI(working_directory=image_folder, option_gui='displayNiftiFiles')
        self.SelectFiles.show()

    def run_DCM2NII(self):
        """wrapper to start the GUI which enables to batch preprocess DICOM dolers and convert them to NIFTI files"""

        # self.convertFiles = MainGuiDcm2nii() # 2021/02/23 not sure if this works without reference to GUI
        # self.convertFiles.show()
        convertFiles2nifti = MainGuiDcm2nii()
        convertFiles2nifti.show()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = GuiTabGeneral()
    w.show()
    sys.exit(app.exec_())
