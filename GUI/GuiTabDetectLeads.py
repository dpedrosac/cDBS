#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QPushButton, QListWidget, QAbstractItemView

import private.allToolTips as setToolTips
import utils.leadManualCorrection as plotElecModel
import utils.preprocLeadCT as LeadDetectionRoutines
from GUI.GuiTwoLists_generic import TwoListGUI
from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration, FileOperations
from utils.settingsLeadDetection import GuiLeadDetection


class GuiTabDetectLeads(QWidget):
    """Tab with options for detecting leads within the (pre-processed and registered) CT imaging"""

    def __init__(self, parent=None):
        super(GuiTabDetectLeads, self).__init__(parent)
        self.selected_subj_ANT = ''

        # General settings/variables/helper files needed at some point
        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['nifti']):
            self.niftidir = self.cfg['folders']['nifti']
        else:
            self.niftidir = FileOperations.set_wdir_in_config(self.cfg, foldername='nifti', init=True)

        self.cfg['folders']['nifti'] = self.niftidir
        self.cfg['folders']['rootdir'] = ROOTDIR
        Configuration.save_config(ROOTDIR, self.cfg)

        self.lay = QHBoxLayout(self)
        self.tab = QWidget()

        # Customize tab
        # ==============================    Tab 3 - Lead detection routines   ==============================
        self.tab.layout = QHBoxLayout()
        self.tab.setLayout(self.tab.layout)

        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab = QGroupBox("Directory")
        self.HBoxUpperLeftTab = QVBoxLayout(self.FolderboxTab)
        self.lblWdirTab = QLabel('wDIR: {}'.format(self.niftidir))
        self.HBoxUpperLeftTab.addWidget(self.lblWdirTab)

        self.btnChangeWdir = QPushButton('Change working directory')
        self.btnChangeWdir.clicked.connect(self.change_wdir)
        self.btnReloadFilesTab = QPushButton('Reload files')
        self.btnReloadFilesTab.clicked.connect(self.run_reload_files)

        self.HBoxUpperLeftTab.addWidget(self.btnChangeWdir)
        self.HBoxUpperLeftTab.addWidget(self.btnReloadFilesTab)

        # ------------------------- Middle left part (Settings)  ------------------------- #
        self.SettingsTabLeadDetect = QGroupBox("Preferences")
        self.HBoxMiddleLeftTabExt = QVBoxLayout(self.SettingsTabLeadDetect)

        self.btn_LeadDetectSettings = QPushButton('Settings \nLead detection')
        self.btn_LeadDetectSettings.clicked.connect(self.run_PreferencesLeadDetection)
        self.btn_LeadDetectSettings.setToolTip(setToolTips.ANTsSettings())

        self.HBoxMiddleLeftTabExt.addWidget(self.btn_LeadDetectSettings)

        # ------------------------- Middle left part (Processing)  ------------------------- #
        self.ActionsTabANTs = QGroupBox("Lead detection routines")
        self.HBoxMiddleLeftTab = QVBoxLayout(self.ActionsTabANTs)
        self.btn_LeadDetectPacer = QPushButton('PaCER algorithm')
        self.btn_LeadDetectPacer.clicked.connect(self.run_LeadDetectionPaCER)

        self.btn_RefineDetectedLeads = QPushButton('Refine detected leads')
        self.btn_RefineDetectedLeads.clicked.connect(self.run_ManualCorrection)

        self.HBoxMiddleLeftTab.addWidget(self.btn_LeadDetectPacer)
        self.HBoxMiddleLeftTab.addWidget(self.btn_RefineDetectedLeads)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.QualityTabLeadDetect = QGroupBox("Quality checks for Lead detection")
        self.HBoxLowerLeftTab = QVBoxLayout(self.QualityTabLeadDetect)
        self.btn_QC_LeadDetect = QPushButton('Check lead detection \nin viewer')
        self.btn_QC_LeadDetect.setToolTip(setToolTips.compareNIFTIfiles())
        self.btn_QC_LeadDetect.clicked.connect(self.VisualiseLeadDetection)
        self.HBoxLowerLeftTab.addWidget(self.btn_QC_LeadDetect)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox = QGroupBox('Available subjects')
        self.HBoxUpperRightTab = QVBoxLayout(self.listbox)
        self.availableNiftiTab = QListWidget()
        self.availableNiftiTab.setSelectionMode(QAbstractItemView.ExtendedSelection)
        itemsTab = FileOperations.list_folders(self.niftidir, prefix=self.cfg['folders']['prefix'])
        self.add_available_items(self.availableNiftiTab, itemsTab, msg='no')
        self.availableNiftiTab.itemSelectionChanged.connect(self.change_list_item)

        self.HBoxUpperRightTab.addWidget(self.availableNiftiTab)

        # Combine all Boxes for Tab 2 Layout
        self.LeftboxTabANTs = QGroupBox()
        self.HBoxTabLeadDetectLeft = QVBoxLayout(self.LeftboxTabANTs)
        self.HBoxTabLeadDetectLeft.addWidget(self.FolderboxTab)
        self.HBoxTabLeadDetectLeft.addStretch(1)
        self.HBoxTabLeadDetectLeft.addWidget(self.SettingsTabLeadDetect)
        self.HBoxTabLeadDetectLeft.addWidget(self.ActionsTabANTs)
        self.HBoxTabLeadDetectLeft.addWidget(self.QualityTabLeadDetect)

        self.tab.layout.addWidget(self.LeftboxTabANTs)
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

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""

        if self.sender() == self.availableNiftiTab:
            items = self.availableNiftiTab.selectedItems()
            self.selected_subj_ANT = []

            for i in range(len(items)):
                self.selected_subj_ANT.append(str(self.availableNiftiTab.selectedItems()[i].text()))

    def add_available_items(self, sending_list, items, msg='yes'):
        """adds the available subjects in the working directory into the items list;
        an error message is dropped if none available"""

        if len(items) == 0 and msg == 'yes':
            buttonReply = QMessageBox.question(self, "No files in directory", "There are no subjects available in "
                                                                              "current working dir: ({}). Change to"
                                                                              " different one?".format(self.niftidir),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.change_wdir()
        else:
            items = list(items)
            items.sort(key=lambda fname: int(fname.split(self.cfg['folders']['prefix'])[1]))
            sending_list.addItems(items)

    def run_reload_files(self):
        """Reloads files, e.g. after renaming them"""
        self.cfg = Configuration.load_config(self.cfg['folders']['rootdir'])
        self.availableNiftiTab.clear()

        itemsChanged = FileOperations.list_folders(self.cfg['folders']['nifti'], prefix=self.cfg['folders']['prefix'])
        self.add_available_items(self.availableNiftiTab, itemsChanged)

    # Separate functions/GUIs that may be initialised here
    def run_LeadDetectionPaCER(self):
        """wrapper to start lead detection with PaCER routines translated to python; original data can be found at:
        https://github.com/adhusch/PaCER/"""

        if len(self.selected_subj_ANT) > 1:
            Output.msg_box(text="Please select only one subject, as multiprocessing for lead detection is not intended",
                           title="Too many subjects selected")
            return
        else:
            msg = "Are you sure you want to process the following subject:\n\n" \
                  "{}".format(''.join(' -> {}\n'.format(c) for c in self.selected_subj_ANT))
            ret = QMessageBox.question(self, 'MessageBox', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                LeadDetectionRoutines.PaCER_script(subjects=self.selected_subj_ANT)

    def run_ManualCorrection(self):
        """wrapper which starts the plotting routine for the detected lead which enables manual corrections"""

        if len(self.selected_subj_ANT) != 1:
            Output.msg_box(text="Please select one and only one subject", title="Subjects selected")
            return
        else:
            msg = "Are you sure you want to process the following subject:\n\n" \
                  "{}".format(''.join(' -> {}\n'.format(c) for c in self.selected_subj_ANT))
            ret = QMessageBox.question(self, 'MessageBox', msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                plotElecModel.PlotRoutines(subject=self.selected_subj_ANT[0],
                                           inputfolder=os.path.join(self.niftidir, self.selected_subj_ANT[0]))

    def run_PreferencesLeadDetection(self):
        """change settings for the Lead detection routines, that is settings for PaCER """
        self.ANTsSettings = GuiLeadDetection()
        self.ANTsSettings.show()

    def VisualiseLeadDetection(self):
        """wrapper to start comparisons between pre- and post-processed images after N4BiasCorrection"""

        if not self.selected_subj_ANT:
            Output.msg_box(text="No folder selected. To proceed, please select at least one.",
                           title="No subject selected")
            return
        elif len(self.selected_subj_ANT) > 1:
            Output.msg_box(text="Please select only one subj.", title="Too many subjects selected")
            return
        else:
            image_folder = os.path.join(self.cfg['folders']['nifti'], self.selected_subj_ANT[0])

        self.SelectFiles = TwoListGUI(working_directory=image_folder, option_gui='displayNiftiFiles')
        self.SelectFiles.show()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = GuiTabDetectLeads()
    w.show()

    sys.exit(app.exec_())
