#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import yaml
from PyQt5 import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QMainWindow, QTabWidget, QListWidget, QAbstractItemView

import utils.HelperFunctions as HF
import utils.preprocANTSpy as ANTspy
from GUI.GUIdcm2nii import MainGuiDcm2nii
from utils.settingsNIFTIprocAnts import GuiSettingsNiftiAnts
from utils.settingsRenameFolders import RenameFolderNames
import private.CheckDefaultFolders as LocationCheck
import private.allToolTips as setToolTips


class ImagingToolboxMain(QMainWindow):
    """ This is the MAIN GUI which enables to create further modules and incorporate them as 'tabs' into the GUI."""

    def __init__(self, _debug=False):
        super().__init__()
        self.setFixedSize(800, 600)
        self.setWindowTitle('Imaging processing pipeline for patients who underwent DBS surgery')
        self.table_widget = TabContent(self, _debug)
        self.setCentralWidget(self.table_widget)
        self.show()


class TabContent(QWidget):
    """creates the different tabs which correspond to the modules which are needed to process data"""
    
    def __init__(self, parent, debug=False, _rootdir="/media/storage/analysis-myoDBS/ImagingToolbox/"):
        super(QWidget, self).__init__(parent)
        self.selected_subj_Gen = ''
        self.selected_subj_ANT = ''

        # General settings/variables/helper files needed needed at some point
        if not _rootdir:
            rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        else:
            rootdir = _rootdir

        self.cfg = HF.LittleHelpers.load_config(rootdir)
        if os.path.isdir(self.cfg["folders"]["nifti"]):
            self.niftidir = self.cfg["folders"]["nifti"]
        else:
            self.niftidir = os.getcwd()
        self.cfg["folders"]["rootdir"] = rootdir
        HF.LittleHelpers.save_config(rootdir, self.cfg)
        self.debug = debug

        # General layout for the tab view and initialisation of tabs
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tabs.addTab(self.tab1, "General")
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "Preprocess Imaging")
        # ----> Further modules/tabs could be added here

        self.tabs.resize(150,100)
        self.layout.addWidget(self.tabs)
        #self.setLayout(self.layout)

        # Customize tabs
        # ==============================    Tab 1 - General   ==============================
        self.tab1.layout = QHBoxLayout()
        self.tab1.setLayout(self.tab1.layout)
        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab1 = QGroupBox("Directory (NIFTI-files)")
        self.HBoxUpperLeftTab1 = QVBoxLayout(self.FolderboxTab1)
        self.lblWdirTab1 = QLabel('wDIR: {}'.format(self.niftidir))
        self.HBoxUpperLeftTab1.addWidget(self.lblWdirTab1)
        self.btnChangeWdir = QPushButton('Change working directory')
        self.btnReloadFilesTab1 = QPushButton('Reload files')
        self.HBoxUpperLeftTab1.addWidget(self.btnChangeWdir)
        self.HBoxUpperLeftTab1.addWidget(self.btnReloadFilesTab1)
        self.btnChangeWdir.clicked.connect(self.change_wdir)
        self.btnReloadFilesTab1.clicked.connect(self.run_reload_files)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.ActionsTab1 = QGroupBox("Functions")
        self.HBoxLowerLeftTab1 = QVBoxLayout(self.ActionsTab1)
        self.btn_demographics = QPushButton('Subject details')
        self.btn_demographics.setToolTip(setToolTips.subjectDetails())
        self.btn_demographics.clicked.connect(self.openDetails)

        self.btn_dcm2nii = QPushButton('Dcm2niix')
        self.btn_dcm2nii.setToolTip(setToolTips.runDCM2NII())
        self.btn_dcm2nii.clicked.connect(self.run_DCM2NII)

        self.btn_viewer = QPushButton('Display\nraw data')
        self.btn_viewer.setToolTip(setToolTips.displayRAWdata())
        self.btn_viewer.clicked.connect(self.view_files)

        self.btn_renaming = QPushButton('Rename\nfolders')
        self.btn_renaming.setToolTip(setToolTips.renameFolders())
        self.btn_renaming.clicked.connect(self.run_rename_folders)

        self.HBoxLowerLeftTab1.addWidget(self.btn_demographics)
        self.HBoxLowerLeftTab1.addWidget(self.btn_dcm2nii)
        self.HBoxLowerLeftTab1.addWidget(self.btn_viewer)
        self.HBoxLowerLeftTab1.addWidget(self.btn_renaming)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox = QGroupBox('Available subjects')
        self.HBoxUpperRightTab1 = QVBoxLayout(self.listbox)
        self.availableNiftiTab1 = QListWidget()
        self.availableNiftiTab1.setSelectionMode(QAbstractItemView.ExtendedSelection)
        itemsTab1 = self.read_subjlist(self.niftidir, prefix=self.cfg["folders"]["prefix"])
        self.availableNiftiTab1.clicked.connect(self.change_list_item)
        # self.availableNiftiTab1.currentItemChanged.connect(self.change_list_item)
        self.add_available_items(self.availableNiftiTab1, itemsTab1)
        self.HBoxUpperRightTab1.addWidget(self.availableNiftiTab1)

        # Combine all Boxes for Tab 1 Layout
        self.LeftboxTab1 = QGroupBox()
        self.HBoxTab1Left = QVBoxLayout(self.LeftboxTab1)
        self.HBoxTab1Left.addWidget(self.FolderboxTab1)
        self.HBoxTab1Left.addStretch()
        self.HBoxTab1Left.addWidget(self.ActionsTab1)

        self.tab1.layout.addWidget(self.LeftboxTab1)
        self.tab1.layout.addWidget(self.listbox)

        # ==============================    Tab 2 - ANTs routines   ==============================
        self.tab2.layout = QHBoxLayout()
        self.tab2.setLayout(self.tab2.layout)
        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab2 = QGroupBox("Directory (NIFTI-files)")
        self.HBoxUpperLeftTab2 = QVBoxLayout(self.FolderboxTab2)
        self.lblWdirTab2 = QLabel('wDIR: {}'.format(self.niftidir))

        self.HBoxUpperLeftTab2.addWidget(self.lblWdirTab2)

        # ------------------------- Middle left part (Settings)  ------------------------- #
        self.SettingsTab2ANTs = QGroupBox("Preferences")
        self.HBoxMiddleLeftTab2ext = QVBoxLayout(self.SettingsTab2ANTs)
        self.btn_ANTsettings = QPushButton('ANT Settings')
        self.btn_ANTsettings.clicked.connect(self.run_ANTsPreferences)
        self.HBoxMiddleLeftTab2ext.addWidget(self.btn_ANTsettings)

        # ------------------------- Middle left part (Processing)  ------------------------- #
        self.ActionsTab2ANTs = QGroupBox("ANTs routines")
        self.HBoxMiddleLeftTab2 = QVBoxLayout(self.ActionsTab2ANTs)
        self.btn_N4BiasCorr = QPushButton('N4BiasCorrect')
        self.btn_MRIreg = QPushButton('MR-Registration')
        self.btn_MRIreg.setToolTip('???')
        self.btn_CTreg = QPushButton('CT-Registration')
        self.btn_N4BiasCorr.clicked.connect(self.run_n4Bias_corr)

        self.HBoxMiddleLeftTab2.addWidget(self.btn_N4BiasCorr)
        self.HBoxMiddleLeftTab2.addWidget(self.btn_MRIreg)
        self.HBoxMiddleLeftTab2.addWidget(self.btn_CTreg)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.QualityTab2ANTs = QGroupBox("Quality checks for ANTs preprocessing")
        self.HBoxLowerLeftTab2 = QVBoxLayout(self.QualityTab2ANTs)
        self.btn_BiasCorrQC = QPushButton('Pre-/Post N4\nBias correction')
        self.btn_BiasCorrQC.setToolTip('???')
        self.btn_RegQC = QPushButton('Pre-/Post \nRegistration')
        #self.btn_RegQC.clicked.connect(self.run_n4Bias_corr)

        self.HBoxLowerLeftTab2.addWidget(self.btn_BiasCorrQC)
        self.HBoxLowerLeftTab2.addWidget(self.btn_RegQC)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox2 = QGroupBox('Available subjects')
        self.HBoxUpperRightTab2 = QVBoxLayout(self.listbox2)
        self.availableNiftiTab2 = QListWidget()
        self.availableNiftiTab2.setSelectionMode(QAbstractItemView.ExtendedSelection)
        itemsTab2 = self.read_subjlist(self.niftidir, prefix=self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab2, itemsTab2, msg='no')
        self.availableNiftiTab2.clicked.connect(self.change_list_item)
        self.HBoxUpperRightTab2.addWidget(self.availableNiftiTab2)

        # Combine all Boxes for Tab 2 Layout
        self.LeftboxTab2 = QGroupBox()
        self.HBoxTab2Left = QVBoxLayout(self.LeftboxTab2)
        self.HBoxTab2Left.addWidget(self.FolderboxTab2)
        self.HBoxTab2Left.addStretch(1)
        self.HBoxTab2Left.addWidget(self.SettingsTab2ANTs)
        self.HBoxTab2Left.addWidget(self.ActionsTab2ANTs)
        self.HBoxTab2Left.addWidget(self.QualityTab2ANTs)

        self.tab2.layout.addWidget(self.LeftboxTab2)
        self.tab2.layout.addWidget(self.listbox2)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def openDetails(self):
        """opens the file which saves the name of the original (DICOM-) folder and the name of the NIFTI-folder after
        conversion in (runDCM2NII) """
        import subprocess

        fileName = os.path.join(self.cfg["folders"]["nifti"], 'subjdetails.csv')
        if os.path.isfile(fileName):
            subprocess.Popen(["xdg-open", ''.join(fileName)])
        else:
            text_nosubj = 'Subject details are not available!'
            title_nosubj = 'Detail file not found'
            HF.LittleHelpers.msg_box(text=text_nosubj, title=title_nosubj)

    def run_DCM2NII(self):
        """wrapper to start the GUI which enables to batch preprocess DICOM dolers and convert them to NIFTI files"""
        self.convertFiles = MainGuiDcm2nii()
        self.convertFiles.show()

    def run_rename_folders(self):
        self.convertFolders = RenameFolderNames()
        self.convertFolders.show()

    def run_reload_files(self):
        """Reloads files, e.g. after renaming them"""
        self.cfg = HF.LittleHelpers.load_config(self.cfg["folders"]["rootdir"])
        self.availableNiftiTab1.clear()
        self.availableNiftiTab2.clear()
        itemsChanged = self.read_subjlist(self.cfg["folders"]["nifti"], prefix=self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab1, itemsChanged)
        self.add_available_items(self.availableNiftiTab2, itemsChanged)

    def run_ANTsPreferences(self):
        """change the settings for the ANTs routines, that is N4BiasCorrection and registration of CT/MRI """
        self.ANTsSettings = GuiSettingsNiftiAnts()
        self.ANTsSettings.show()

    def run_n4Bias_corr(self):
        """wrapper to start the preprocessing, that is the GUI in which the different options for ANTs routines
        are displayed"""

        if not self.selected_subj_ANT:
            HF.LittleHelpers.msg_box(text="No folder selected. To proceed, please indicate what folder to process. "
                                          "(For this option, numerous folders are possible for batch processing)",
                                     title="No subject selected")
        else:
            msg = "Are you sure you want to process all NIFTI-files in the following folders:\n" \
                  "{}".format(''.join('- {}\n'.format(c) for c in sorted(self.selected_subj_ANT)))
            ret = QMessageBox.question(self, 'MessageBox', msg,
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                ANTspy.ProcessANTSpy().N4BiasCorrection(subjects=self.selected_subj_ANT)
        #self.convertFiles = MainGUIDCM2NII()
        #self.convertFiles.show()

    @staticmethod   # TODO move this to helper functions???
    def read_subjlist(inputdir, prefix='subj', files2lookfor='NIFTI'):
        """takes folder and lists all available subjects in this folder according to some filter given as [prefix]"""

        list_all = [name for name in os.listdir(inputdir)
                    if (os.path.isdir(os.path.join(inputdir, name)) and prefix in name)]

        if list_all == '':
            list_subj = 'No available subjects, please make sure {}-files are present and correct ' \
                        '"prefix" is given'.format(files2lookfor)
        else:
            #list_subj = [x.split("_")[0] for x in list_all]
            list_subj = set(list_all)

        return list_subj

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
            sending_list.addItems(items)

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""
        
        if self.sender() == self.availableNiftiTab1:
            items = self.availableNiftiTab1.selectedItems()
            self.selected_subj_Gen = []

            for i in range(len(items)):
                self.selected_subj_Gen.append(str(self.availableNiftiTab1.selectedItems()[i].text()))
#            print(self.selected_subj_Gen)
        elif self.sender() == self.availableNiftiTab2:
            items = self.availableNiftiTab2.selectedItems()
            self.selected_subj_ANT = []

            for i in range(len(items)):
                self.selected_subj_ANT.append(str(self.availableNiftiTab2.selectedItems()[i].text()))

            #self.selected_subj_ANT = self.availableNiftiTab2.currentItem().text()

    def change_wdir(self):
        """A new window appears in which the working directory can be set; if set, this is stored in the configuration
        file, so that upon the next start there is the same folder selected automatically"""
        self.niftidir = QFileDialog.getExistingDirectory(self, 'title')

        self.cfg["folders"]["nifti"] = self.niftidir
        with open(os.path.join(os.getcwd(), 'config_imagingTB.yaml'), 'wb') as settings_mod:
            yaml.safe_dump(self.cfg, settings_mod, default_flow_style=False,
                           explicit_start=True, allow_unicode=True, encoding='utf-8')

        if self.debug:
            print(self.niftidir)
        self.lblWdirTab1.setText('wDIR: {}'.format(self.niftidir))
        self.availableNiftiTab1.clear()
        self.availableNiftiTab2.clear()

        itemsChanged = self.read_subjlist(self.cfg["folders"]["nifti"], self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab1, itemsChanged)
        self.add_available_items(self.availableNiftiTab2, itemsChanged, msg='no')

    def view_files(self):
        """this function displays the selected folder/subject in an external viewer (default:ITK-snap)"""
        if not self.selected_subj_Gen:
            HF.LittleHelpers.msg_box(text="No folder selected. To proceed, please indicate which subject to look for",
                                     title="No subject selected")
            return
        elif len(self.selected_subj_Gen) > 1:
            HF.LittleHelpers.msg_box(text="Please select only one folder to avoid loading too many images",
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

        #image_filename.append("d:/analysis-myoDBS/data/NIFTI/subj1/t2_spc_FoV256_1iso_12ch_t2_spc_FoV256_1iso_12ch.nii")
        HF.LittleHelpers.load_imageviewer(viewer, image_folder)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ImagingToolboxMain()
    sys.exit(app.exec_())
