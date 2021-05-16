#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QPushButton, QListWidget, QAbstractItemView

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration, FileOperations, Imaging

# TODO: 1.) create ToolTips for the entire script, 2.) Modify SST creation with a) warning, b) check for requisites,
#  and c) creation of shell-script

class GuiTabTemplate(QWidget):
    """General tab which enables import of DICOM files but also a set of distinct options such as viewing the
    metadata or displaying images in an external viewer and renaming folders"""

    def __init__(self, parent=None):
        super(GuiTabTemplate, self).__init__(parent)
        self.selected_subj_Gen = ''
        self.wdirTemplate = os.path.join(ROOTDIR, 'ext', 'templates')

        # General settings/variables/helper files needed needed at some point
        self.cfg = Configuration.load_config(ROOTDIR)
        if os.path.isdir(self.cfg['folders']['nifti']):
            self.niftidir = self.cfg['folders']['nifti']
        else:
            self.niftidir = FileOperations.set_wdir_in_config(self.cfg, foldername='nifti', init=True)
        self.cfg['folders']['rootdir'] = ROOTDIR
        Configuration.save_config(ROOTDIR, self.cfg)

        self.lay = QHBoxLayout(self)
        self.tab = QWidget()

        # Customize tab
        # ==============================    Tab 1 - General   ==============================
        self.tab.layout = QHBoxLayout()
        self.tab.setLayout(self.tab.layout)
        # ------------------------- Upper left part (Folder)  ------------------------- #
        self.FolderboxTab = QGroupBox("Directory (Templates)")
        self.HBoxUpperLeftTab = QVBoxLayout(self.FolderboxTab)

        self.dirTemplates = QLabel('wDIR: {}'.format(self.wdirTemplate))
        self.HBoxUpperLeftTab.addWidget(self.dirTemplates)

        self.btnChangeWdir = QPushButton('Change working directory')
        self.btnChangeWdir.setDisabled(True)

        self.btnReloadFilesTab = QPushButton('Reload files')
        self.btnReloadFilesTab.clicked.connect(self.run_reload_files)

        self.HBoxUpperLeftTab.addWidget(self.btnChangeWdir)
        self.HBoxUpperLeftTab.addWidget(self.btnReloadFilesTab)

        # ------------------------- Lower left part (Processing)  ------------------------- #
        self.ActionsTab = QGroupBox("Functions")
        self.HBoxLowerLeftTab = QVBoxLayout(self.ActionsTab)
        self.btn_new_default = QPushButton('Set new \ndefault')
        # self.btn_subj_details.setToolTip(setToolTips.subjectDetails()) TODO: new ToolTip needed
        self.btn_new_default.clicked.connect(self.redefineDefault)

        self.btn_viewer = QPushButton('View selected \nTemplate in viewer')
        # self.btn_viewer.setToolTip(setToolTips.displayFolderContent()) TODO: new ToolTip needed
        self.btn_viewer.clicked.connect(self.view_template)

        self.create_SST = QPushButton('Create Study-specific\ntemplate')
        # self.btn_renaming.setToolTip(setToolTips.renameFolders()) TODO: new ToolTip needed
        self.create_SST.clicked.connect(self.create_StudySpecificTemplate)

        self.HBoxLowerLeftTab.addWidget(self.btn_viewer)
        self.HBoxLowerLeftTab.addWidget(self.btn_new_default)
        self.HBoxLowerLeftTab.addWidget(self.create_SST)

        # -------------------- Right part (Subject list)  ----------------------- #
        self.listbox = QGroupBox('Available subjects')
        self.HBoxUpperRightTab = QVBoxLayout(self.listbox)
        self.availableTemplates = QListWidget()
        self.availableTemplates.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.availableTemplates.itemSelectionChanged.connect(self.change_list_item)
        itemsTab = set(FileOperations.list_files_in_folder(self.wdirTemplate))
        self.add_available_templates(self.availableTemplates, itemsTab)
        self.HBoxUpperRightTab.addWidget(self.availableTemplates)

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
    def run_reload_files(self):
        """Reloads files, e.g. after renaming them"""

        self.cfg = Configuration.load_config(self.cfg['folders']['rootdir'])
        self.availableTemplates.clear()
        itemsTab = set(FileOperations.list_files_in_folder(self.wdirTemplate))
        self.add_available_templates(self.availableTemplates, itemsTab)

    def change_list_item(self):
        """function intended to provide the item which is selected. As different tabs have a similar functioning, it is
         coded in a way that the sender is identified"""

        if self.sender() == self.availableTemplates:
            items = self.availableTemplates.selectedItems()
            self.selected_subj_Gen = []

            for i in range(len(items)):
                self.selected_subj_Gen.append(str(self.availableTemplates.selectedItems()[i].text()))

    def add_available_templates(self, sending_list, items, msg="yes"):
        """adds the available templates in the working directory into the items list;
        an error message is dropped if none available"""
        if len(items) == 0 and msg == "yes":
            buttonReply = QMessageBox.question(self, "No templates in folder", "There are no templates available "
                                               "in the directory: {}. Please make sure that at least the default "
                                               "ones are in this directory! Retry?".format(os.path.join(ROOTDIR, 'ext',
                                                                                                        'templates')),
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if buttonReply == QMessageBox.Yes:
                self.run_reload_files()
        else:
            sending_list.addItems(list(items))
            myFont = QtGui.QFont()
            myFont.setItalic(True)
            out = sending_list.findItems(os.path.basename(self.cfg['folders']['default_template']),
                                         QtCore.Qt.MatchExactly)
            out[0].setFont(myFont)

    def redefineDefault(self):
        """redefines which template is set as default on the right list and in the cfg-file"""

        if not self.selected_subj_Gen:
            Output.msg_box(text="No template selected. Please indicate new default one.", title="No data selected")
            return
        elif len(self.selected_subj_Gen) > 1:
            Output.msg_box(text="Please select only one image as default.", title="Too many templates selected")
            return
        else:
            default_template = [FileOperations.return_full_filename(self.wdirTemplate, x) for x in self.selected_subj_Gen]
            self.cfg['folders']['default_template'] = default_template[0]
            Configuration.save_config(ROOTDIR, self.cfg)

        self.run_reload_files()

    def create_StudySpecificTemplate(self):
        """Renames all folders with a similar prefix; After that manual reloading is necessary"""
        print('not yet implemented')

    def view_template(self):
        """this function opens a list dialog and enables selecting NIFTI files for e.g. check the content (identical
        function as in GUITabPreprocessANTs.py."""

        if not self.selected_subj_Gen:
            Output.msg_box(text="No image selected. Please indicate image(s) to load.", title="No templates selected")
            return
        else:
            template_list = [FileOperations.return_full_filename(self.wdirTemplate, x) for x in self.selected_subj_Gen]
            Imaging.load_imageviewer('itk-snap', template_list)  # to-date, only itk-snap available. could be changed


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = GuiTabTemplate()
    w.show()
    sys.exit(app.exec_())
