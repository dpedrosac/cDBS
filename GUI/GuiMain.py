#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QMainWindow, QTabWidget

from GUI.GuiTabDetectLeads import GuiTabDetectLeads
from GUI.GuiTabGeneral import GuiTabGeneral
from GUI.GuiTabTemplate import GuiTabTemplate
from GUI.GuiTabPreprocessANTs import GuiTabPreprocessANTs
from dependencies import ROOTDIR
from utils.HelperFunctions import Configuration


class cDBSMain(QMainWindow):
    """ This is the MAIN GUI which enables to create further modules and incorporate them as 'tabs' into the GUI."""

    def __init__(self, _debug=False):
        super().__init__()
        #self.setFixedSize(800, 600)
        self.setWindowTitle('Imaging processing pipeline for patients who underwent DBS surgery')
        self.table_widget = TabContent(self)
        self.setCentralWidget(self.table_widget)
        self.show()


class TabContent(QWidget):
    """creates the different tabs which correspond to the modules which are needed to process data"""

    def __init__(self, parent=None):
        super(TabContent, self).__init__()
        self.selected_subj_Gen = ''
        self.selected_subj_ANT = ''

        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['nifti']):
            self.niftidir = self.cfg['folders']['nifti']
        else:
            self.niftidir = os.getcwd()
        self.cfg['folders']['rootdir'] = ROOTDIR
        Configuration.save_config(ROOTDIR, self.cfg)

        # General layout for the tab view and initialisation of tabs
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.resize(150, 100)
        self.layout.addWidget(self.tabs)

        # Customize tabs
        # ==============================    Tab 1 - General   ==============================
        self.tab1 = GuiTabGeneral()
        self.tabs.addTab(self.tab1, "General")

        # ==============================    Tab 2 - Work on templates   ==============================
        self.tab2 = GuiTabTemplate()
        self.tabs.addTab(self.tab2, "Template")

        # ==============================    Tab 3 - ANTs routines   ==============================
        self.tab3 = GuiTabPreprocessANTs()
        self.tabs.addTab(self.tab3, "Preprocess Imaging (ANTsPy)")

        # ==============================    Tab 3 - Lead detection   ==============================
        self.tab4 = GuiTabDetectLeads()
        self.tabs.addTab(self.tab4, "Detect leads (Pacer)")

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = cDBSMain()
    sys.exit(app.exec_())
