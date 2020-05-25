#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QListWidget, QAbstractItemView

import utils.HelperFunctions as HF
from GUI.GUIdcm2nii import MainGuiDcm2nii
from utils.settingsRenameFolders import RenameFolderNames
import private.CheckDefaultFolders as LocationCheck
import private.allToolTips as setToolTips


class GuiTabGeneral(QWidget):
    """General tab which enables import of DICOM files but also a set of distinct options such as viewing the
    metadata or displaying images in an external viewer and renaming folders"""

    def __init__(self, parent=None, ROOTDIR=''):
        super(GuiTabGeneral, self).__init__(parent)
        self.selected_subj_Gen = ''

        # General settings/variables/helper files needed needed at some point
        if not ROOTDIR:
            ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)
        if os.path.isdir(self.cfg["folders"]["nifti"]):
            self.niftidir = self.cfg["folders"]["nifti"]
        else:
            self.niftidir = os.getcwd()
        self.cfg["folders"]["rootdir"] = ROOTDIR
        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)

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
        self.btnReloadFilesTab = QPushButton('Reload files')
        self.HBoxUpperLeftTab.addWidget(self.btnChangeWdir)
        self.HBoxUpperLeftTab.addWidget(self.btnReloadFilesTab)
        self.btnChangeWdir.clicked.connect(self.change_wdir)
        self.btnReloadFilesTab.clicked.connect(self.run_reload_files)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.ActionsTab = QGroupBox("Functions")
        self.HBoxLowerLeftTab = QVBoxLayout(self.ActionsTab)
        self.btn_subj_details = QPushButton('Subject details')
        self.btn_subj_details.setToolTip(setToolTips.subjectDetails())
        self.btn_subj_details.clicked.connect(self.openDetails)

        self.btn_dcm2nii = QPushButton('Dcm2niix')
        self.btn_dcm2nii.setToolTip(setToolTips.runDCM2NII())
        self.btn_dcm2nii.clicked.connect(self.run_DCM2NII)

        self.btn_viewer = QPushButton('Display\nraw data')
        self.btn_viewer.setToolTip(setToolTips.displayRAWdata())
        self.btn_viewer.clicked.connect(self.view_files)

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

        itemsTab = HF.read_subjlist(self.niftidir, prefix=self.cfg["folders"]["prefix"])
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

    # Start with the functions for lists and buttons in this tab
    def change_wdir(self):
        """A new window appears in which the working directory for NIFTI-files can be set; if set, this is stored
         in the configuration file, so that upon the next start there is the same folder selected automatically"""

        self.niftidir = QFileDialog.getExistingDirectory(self, 'Please select the directory of nii-files')
        self.lblWdirTab.setText('wDIR: {}'.format(self.niftidir))

        self.cfg["folders"]["nifti"] = self.niftidir
        with open(os.path.join(os.getcwd(), 'config_imagingTB.yaml'), 'wb') as settings_mod:
            yaml.safe_dump(self.cfg, settings_mod, default_flow_style=False,
                           explicit_start=True, allow_unicode=True, encoding='utf-8')

        self.availableNiftiTab.clear()
        itemsChanged = HF.read_subjlist(self.cfg["folders"]["nifti"], self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab, itemsChanged)

    def run_reload_files(self):
        """Reloads files, e.g. after renaming them"""
        self.cfg = HF.LittleHelpers.load_config(self.cfg["folders"]["rootdir"])
        self.availableNiftiTab.clear()
        #self.availableNiftiTab2.clear()
        itemsChanged = HF.read_subjlist(self.cfg["folders"]["nifti"], prefix=self.cfg["folders"]["prefix"])
#        itemsChanged = self.read_subjlist(self.cfg["folders"]["nifti"], prefix=self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab, itemsChanged)
        #self.add_available_items(self.availableNiftiTab2, itemsChanged)

# This was moved to helper functions. Can be renmoved if there is no further problem in a future
#    @staticmethod
#    def read_subjlist(inputdir, prefix='subj', files2lookfor='NIFTI'):
#        """takes folder and lists all available subjects in this folder according to some filter given as [prefix]"""
#
#        list_all = [name for name in os.listdir(inputdir)
#                    if (os.path.isdir(os.path.join(inputdir, name)) and prefix in name)]
#
#        if list_all == '':
#            list_subj = 'No available subjects, please make sure {}-files are present and correct ' \
#                        '"prefix" is given'.format(files2lookfor)
#        else:
#            #list_subj = [x.split("_")[0] for x in list_all]
#            list_subj = set(list_all)
#
#        return list_subj

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""

        if self.sender() == self.availableNiftiTab:
            items = self.availableNiftiTab.selectedItems()
            self.selected_subj_Gen = []

            for i in range(len(items)):
                self.selected_subj_Gen.append(str(self.availableNiftiTab.selectedItems()[i].text()))
        #            print(self.selected_subj_Gen)
        #elif self.sender() == self.availableNiftiTab2:
        #    items = self.availableNiftiTab2.selectedItems()
        #    self.selected_subj_ANT = []

        #    for i in range(len(items)):
        #        self.selected_subj_ANT.append(str(self.availableNiftiTab2.selectedItems()[i].text()))

            # self.selected_subj_ANT = self.availableNiftiTab2.currentItem().text()

    def add_available_items(self, sending_list, items, msg='yes'):
        """adds the available subjects in the working directory into the items list;
        an error message is dropped if none available"""
        if len(items) == 0 and msg == 'yes':
            buttonReply = QMessageBox.question(self, 'No files in dir', 'There are no subjects available '
                                               'in the current working directory ({}). Do you want to '
                                               ' change to a different one?'.format(self.niftidir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_wdir()
        else:
            items = list(items)
            items.sort(key=lambda fname: int(fname.split(self.cfg["folders"]["prefix"])[1]))
            sending_list.addItems(items)

    # Separate functions/GUIs that may be initialised here
    def openDetails(self):
        """opens details file which has additional information on subjects """
        import subprocess

        fileName = os.path.join(self.cfg["folders"]["nifti"], 'subjdetails.csv')
        if os.path.isfile(fileName):
            subprocess.Popen(["xdg-open", ''.join(fileName)])
        else:
            HF.msg_box(text='Subject details are not available!', title='Detail file not found')


    def run_rename_folders(self):
        from dependencies import ROOTDIR
        self.convertFolders = RenameFolderNames(ROOTDIR)
        self.convertFolders.show()

    def view_files(self):
        """this function displays the selected folder/subject in an external viewer (default:ITK-snap)"""
        if not self.selected_subj_Gen:
            HF.msg_box(text="No folder selected. To proceed, please indicate which subject to look for",
                                     title="No subject selected")
            return
        elif len(self.selected_subj_Gen) > 1:
            HF.msg_box(text="Please select only one folder to avoid loading too many images",
                                     title="Too many subjects selected")
            return
        else:
            image_folder = os.path.join(self.cfg["folders"]["nifti"], self.selected_subj_Gen[0])

        viewer = 'itk-snap' # to-date, only one viewer is available. May be changed in a future
        if not self.cfg["folders"]["path2itksnap"]:
            self.cfg["folders"]["path2itksnap"] = \
                LocationCheck.FileLocation.itk_snap_check(self.cfg["folders"]["rootdir"])
            HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)
            viewer_path = self.cfg["folders"]["path2itksnap"]

        #if viewer == 'slicer':
        #    if not os.path.isdir(self.cfg["path2slicer
        #        self.path2viwewer = QFileDialog.getExistingDirectory(self, 'Please indicate location of 3D slicer.')
        #        self.cfg["path2slicer"] = self.path2viwewer
        #    path2viewer = self.cfg["path2slicer"]

        # TODO code for image viewer was changed so that instead of a folder the images are needed as input (list)
        print("Code need s to be updated. Stopping here!!")
        #HF.LittleHelpers.load_imageviewer(viewer, image_folder)

    def run_DCM2NII(self):
        """wrapper to start the GUI which enables to batch preprocess DICOM dolers and convert them to NIFTI files"""
        self.convertFiles = MainGuiDcm2nii()
        self.convertFiles.show()



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = GuiTabGeneral()
    w.show()
    sys.exit(app.exec_())