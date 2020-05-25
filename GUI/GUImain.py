#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import yaml
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QFileDialog, QPushButton, QMainWindow, QTabWidget, QListWidget, QAbstractItemView

import utils.HelperFunctions as HF
import utils.preprocANTSpy as ANTspy
from GUI.GuiTabGeneral import GuiTabGeneral
from GUI.GuiTabPrepreocessANTs import GuiTabPreprocessANTs
from GUI.GUIdcm2nii import MainGuiDcm2nii
from utils.settingsNIFTIprocAnts import GuiSettingsNiftiAnts
from utils.settingsRenameFolders import RenameFolderNames
import private.CheckDefaultFolders as LocationCheck
from dependencies import ROOTDIR
import private.allToolTips as setToolTips


class cDBSMain(QMainWindow):
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

    def __init__(self, parent=None, debug=False):
        super(TabContent, self).__init__()
        self.selected_subj_Gen = ''
        self.selected_subj_ANT = ''

        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)
        if os.path.isdir(self.cfg["folders"]["nifti"]):
            self.niftidir = self.cfg["folders"]["nifti"]
        else:
            self.niftidir = os.getcwd()
        self.cfg["folders"]["rootdir"] = ROOTDIR
        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)
        self.debug = debug

        # General layout for the tab view and initialisation of tabs
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.resize(150, 100)
        self.layout.addWidget(self.tabs)
        # self.setLayout(self.layout)

        # Customize tabs
        # ==============================    Tab 1 - General   ==============================
        self.tab1 = GuiTabGeneral()
        self.tabs.addTab(self.tab1, "General")

        # ==============================    Tab 2 - ANTs routines   ==============================
        self.tab2 = GuiTabPreprocessANTs()
        self.tabs.addTab(self.tab2, "Preprocess Imaging (ANTsPy)")

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = cDBSMain()
    sys.exit(app.exec_())
