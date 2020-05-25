#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import yaml
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QListWidget, QAbstractItemView

import utils.HelperFunctions as HF
import utils.preprocANTSpy as ANTspy
from utils.settingsNIFTIprocAnts import GuiSettingsNiftiAnts


class GuiTabPreprocessANTs(QWidget):
    """Tab which shows the options for preprocessing data, that is N4BiasfieldCorrection and Coregestiering of pre- and
     postoperative imaging"""

    def __init__(self, parent=None, ROOTDIR=''):
        super(GuiTabPreprocessANTs, self).__init__(parent)
        self.selected_subj_ANT = ''

        # General settings/variables/helper files needed needed at some point
        if not ROOTDIR:
            from dependencies import ROOTDIR

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
        # ==============================    Tab 2 - ANTs routines   ==============================
        self.tab.layout = QHBoxLayout()
        self.tab.setLayout(self.tab.layout)

        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab = QGroupBox("Directory (NIFTI-files)")
        self.HBoxUpperLeftTab = QVBoxLayout(self.FolderboxTab)
        self.lblWdirTab = QLabel('wDIR: {}'.format(self.niftidir))
        self.HBoxUpperLeftTab.addWidget(self.lblWdirTab)
        self.btnChangeWdir = QPushButton('Change working directory')
        self.HBoxUpperLeftTab.addWidget(self.btnChangeWdir)
        self.btnChangeWdir.clicked.connect(self.change_wdir)

        # ------------------------- Middle left part (Settings)  ------------------------- #
        self.SettingsTabANTs = QGroupBox("Preferences")
        self.HBoxMiddleLeftTabExt = QVBoxLayout(self.SettingsTabANTs)
        self.btn_ANTsettings = QPushButton('ANT Settings')
        self.btn_ANTsettings.clicked.connect(self.run_ANTsPreferences)
        self.HBoxMiddleLeftTabExt.addWidget(self.btn_ANTsettings)

        # ------------------------- Middle left part (Processing)  ------------------------- #
        self.ActionsTabANTs = QGroupBox("ANTs routines")
        self.HBoxMiddleLeftTab = QVBoxLayout(self.ActionsTabANTs)
        self.btn_N4BiasCorr = QPushButton('N4BiasCorrect')
        self.btn_MRIreg = QPushButton('MR-Registration')
        self.btn_MRIreg.setToolTip('???')
        self.btn_CTreg = QPushButton('CT-Registration')
        self.btn_N4BiasCorr.clicked.connect(self.run_n4Bias_corr)

        self.HBoxMiddleLeftTab.addWidget(self.btn_N4BiasCorr)
        self.HBoxMiddleLeftTab.addWidget(self.btn_MRIreg)
        self.HBoxMiddleLeftTab.addWidget(self.btn_CTreg)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.QualityTabANTs = QGroupBox("Quality checks for ANTs preprocessing")
        self.HBoxLowerLeftTab = QVBoxLayout(self.QualityTabANTs)
        self.btn_BiasCorrQC = QPushButton('Pre-/Post N4\nBias correction')
        self.btn_BiasCorrQC.setToolTip('???')
        self.btn_RegQC = QPushButton('Pre-/Post \nRegistration')
        #self.btn_RegQC.clicked.connect(self.run_n4Bias_corr)

        self.HBoxLowerLeftTab.addWidget(self.btn_BiasCorrQC)
        self.HBoxLowerLeftTab.addWidget(self.btn_RegQC)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox = QGroupBox('Available subjects')
        self.HBoxUpperRightTab = QVBoxLayout(self.listbox)
        self.availableNiftiTab = QListWidget()
        self.availableNiftiTab.setSelectionMode(QAbstractItemView.ExtendedSelection)
        itemsTab = HF.read_subjlist(self.niftidir, prefix=self.cfg["folders"]["prefix"])
        self.add_available_items(self.availableNiftiTab, itemsTab, msg='no')
        self.availableNiftiTab.itemSelectionChanged.connect(self.change_list_item)

        self.HBoxUpperRightTab.addWidget(self.availableNiftiTab)

        # Combine all Boxes for Tab 2 Layout
        self.LeftboxTabANTs = QGroupBox()
        self.HBoxTabANTsLeft = QVBoxLayout(self.LeftboxTabANTs)
        self.HBoxTabANTsLeft.addWidget(self.FolderboxTab)
        self.HBoxTabANTsLeft.addStretch(1)
        self.HBoxTabANTsLeft.addWidget(self.SettingsTabANTs)
        self.HBoxTabANTsLeft.addWidget(self.ActionsTabANTs)
        self.HBoxTabANTsLeft.addWidget(self.QualityTabANTs)

        self.tab.layout.addWidget(self.LeftboxTabANTs)
        self.tab.layout.addWidget(self.listbox)

        # Add tabs to widget
#        self.layout.addWidget(self.tabs)
#        self.setLayout(self.layout)

        self.lay.addWidget(self.tab)

    # ------------------------- Start with the functions for lists and buttons in this tab  ------------------------- #
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

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""

        if self.sender() == self.availableNiftiTab:
            items = self.availableNiftiTab.selectedItems()
            self.selected_subj_ANT = []

            for i in range(len(items)):
                self.selected_subj_ANT.append(str(self.availableNiftiTab.selectedItems()[i].text()))
        #            print(self.selected_subj_Gen)

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
    def run_n4Bias_corr(self):
        """wrapper to start the preprocessing, that is the GUI in which the different options for ANTs routines
        are displayed"""

        if not self.selected_subj_ANT:
            HF.msg_box(text="No folder selected. To proceed, please indicate what folder to process. "
                                          "(For this option, numerous folders are possible for batch processing)",
                                     title="No subject selected")
        else:
            msg = "Are you sure you want to process all NIFTI-files in the following folders:\n\n" \
                  "{}".format(''.join(' -> {}\n'.format(c) for c in self.selected_subj_ANT))
            ret = QMessageBox.question(self, 'MessageBox', msg,
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                ANTspy.ProcessANTSpy().N4BiasCorrection(subjects=self.selected_subj_ANT)

    def run_ANTsPreferences(self):
        """change the settings for the ANTs routines, that is N4BiasCorrection and registration of CT/MRI """
        self.ANTsSettings = GuiSettingsNiftiAnts()
        self.ANTsSettings.show()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = GuiTabPreprocessANTs()
    w.show()

    sys.exit(app.exec_())
