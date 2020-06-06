#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import yaml
import os
import utils.HelperFunctions as HF
import private.allToolTips as setToolTips
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, \
    QPushButton, QButtonGroup, QRadioButton, QLineEdit, QMessageBox
from dependencies import ROOTDIR

class GuiSettingsNiftiAnts(QWidget):
    """ Helper GUI, which enables to change the settings of the different steps with the ANts Toolbox to analyse NIFTI
      data"""

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent=None)

        # Load configuration files and general settings
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)

        # General appearance of the GUI
        self.setFixedSize(800, 600)
        self.setWindowTitle('Settings for working with NIFTI files and ANTS Toolbox')
        self.show()

        # Create general layout
        self.layout_tot = QVBoxLayout(self)

        # ==============================    Create Content for Left Upper Box   ==============================
        self.optionboxBias = QGroupBox('N4BiasCorrection')
        self.settings_list1 = QVBoxLayout(self.optionboxBias)

        self.labelPrefix = QLabel('Output prefix?\t\t')
        self.labelPrefix.setToolTip(setToolTips.LabelPrefixBias())
        self.lineEditPrefix = QLineEdit()

        lay1 = QHBoxLayout()
        lay1.addWidget(self.labelPrefix)
        lay1.addWidget(self.lineEditPrefix)
        lay1.addStretch()

        self.labelDenoise = QLabel('Denoise images?\t\t')
        #        self.labelPrefix.setToolTip(setToolTips.LabelPrefixBias())
        self.btngroup_Denoise = QButtonGroup()
        self.rbtnDenoisey = QRadioButton('yes')
        self.rbtnDenoisen = QRadioButton('no')
        self.rbtnDenoisey.toggled.connect(self.onClickedRBTN_Denoise)
        self.rbtnDenoisen.toggled.connect(self.onClickedRBTN_Denoise)

        self.btngroup_Denoise.addButton(self.rbtnDenoisey)
        self.btngroup_Denoise.addButton(self.rbtnDenoisen)
        lay2 = QHBoxLayout()
        lay2.addWidget(self.labelDenoise)
        lay2.addWidget(self.rbtnDenoisey)
        lay2.addWidget(self.rbtnDenoisen)
        lay2.addStretch()

        self.labelShrink = QLabel('Shrink factor?\t\t')
        self.labelShrink.setToolTip(setToolTips.LabelShrink())
        self.lineEditShrink = QLineEdit()
        regex = QtCore.QRegExp('^[0-4]\d{0}$')
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditShrink.setValidator(validator1)

        lay3 = QHBoxLayout()
        lay3.addWidget(self.labelShrink)
        lay3.addWidget(self.lineEditShrink)
        lay3.addStretch()

        # TODO double check the Label and the ToolTip to ensure accuracy
        self.labelBSplineDist = QLabel('Distance BSplines?\t')
        self.labelBSplineDist.setToolTip(setToolTips.BSplineDistance())
        self.lineEditBSplineDist = QLineEdit()
        regex = QtCore.QRegExp('^[0-9]\d{2}$')
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditBSplineDist.setValidator(validator1)

        lay4 = QHBoxLayout()
        lay4.addWidget(self.labelBSplineDist)
        lay4.addWidget(self.lineEditBSplineDist)
        lay4.addStretch()

        # TODO: Change labelConvergence
        width = 31.33
        self.labelConv = QLabel('Convergence?\t\t')
        self.labelConv.setToolTip(setToolTips.N4BiasConvergence())
        self.lineEditConv1 = QLineEdit()
        self.lineEditConv1.setFixedWidth(width)
        self.lineEditConv2 = QLineEdit()
        self.lineEditConv2.setFixedWidth(width)
        self.lineEditConv3 = QLineEdit()
        self.lineEditConv3.setFixedWidth(width)
        self.lineEditConv4 = QLineEdit()
        self.lineEditConv4.setFixedWidth(width)

        lay5 = QHBoxLayout()
        lay5.addWidget(self.labelConv)
        lay5.addWidget(self.lineEditConv1)
        lay5.addWidget(self.lineEditConv2)
        lay5.addWidget(self.lineEditConv3)
        lay5.addWidget(self.lineEditConv4)
        lay5.addStretch()

        self.labelTolerance = QLabel('Tolerance?\t\t')
        self.labelTolerance.setToolTip(setToolTips.N4BiasConvergence())
        self.lineEditTolerance = QLineEdit()

        lay6 = QHBoxLayout()
        lay6.addWidget(self.labelTolerance)
        lay6.addWidget(self.lineEditTolerance)
        lay6.addStretch()

        self.labelDiffPrefix = QLabel('Prefix for DTI data?\t')
        self.labelDiffPrefix.setToolTip(setToolTips.DiffPrefix())
        self.lineEditDiffPrefix = QLineEdit()

        lay7 = QHBoxLayout()
        lay7.addWidget(self.labelDiffPrefix)
        lay7.addWidget(self.lineEditDiffPrefix)
        lay7.addStretch()

        self.settings_list1.addLayout(lay1)
        self.settings_list1.addLayout(lay2)
        self.settings_list1.addLayout(lay3)
        self.settings_list1.addLayout(lay4)
        self.settings_list1.addLayout(lay5)
        self.settings_list1.addLayout(lay6)
        self.settings_list1.addLayout(lay7)
        self.settings_list1.addStretch()

        # ==============================    Create Content for Right Upper Box   ==============================
        self.optionboxRegistration = QGroupBox('ImageRegistration')
        self.settings_list2 = QVBoxLayout(self.optionboxRegistration)
        self.settings_list2.addLayout(lay1)

        self.labelPrefixRegistration = QLabel('Output prefix?\t\t')
        #self.labelPrefixRegistration.setToolTip(setToolTips.LabelPrefixBias())
        self.lineEditPrefixRegistration = QLineEdit()

        lay8 = QHBoxLayout()
        lay8.addWidget(self.labelPrefixRegistration)
        lay8.addWidget(self.lineEditPrefixRegistration)
        lay8.addStretch()

        self.labelResampleSpacing = QLabel('Resample Spacing?\t')
        self.labelResampleSpacing.setToolTip(setToolTips.LabelResampleImages())
        self.lineResampleSpacing = QLineEdit()

        lay9 = QHBoxLayout()
        lay9.addWidget(self.labelResampleSpacing)
        lay9.addWidget(self.lineResampleSpacing)
        lay9.addStretch()

        self.labelResampleMethod = QLabel('Resampling method?\t')
        self.labelResampleMethod.setToolTip(setToolTips.ResampleMethod())
        self.btngroup_ResampleMethod = QButtonGroup()
        self.rbtnResample0 = QRadioButton("0") #one of 0 (linear), 1 (nearest neighbor), 2 (gaussian), 3 (windowed sinc), 4 (bspline)
        self.rbtnResample1 = QRadioButton("1")
        self.rbtnResample2 = QRadioButton("2")
        self.rbtnResample3 = QRadioButton("3")
        self.rbtnResample4 = QRadioButton("4")
        self.rbtnResample0.toggled.connect(self.onClickedRBTN_ResampleMethod)
        self.rbtnResample1.toggled.connect(self.onClickedRBTN_ResampleMethod)
        self.rbtnResample2.toggled.connect(self.onClickedRBTN_ResampleMethod)
        self.rbtnResample3.toggled.connect(self.onClickedRBTN_ResampleMethod)
        self.rbtnResample4.toggled.connect(self.onClickedRBTN_ResampleMethod)

        self.btngroup_ResampleMethod.addButton(self.rbtnResample0)
        self.btngroup_ResampleMethod.addButton(self.rbtnResample1)
        self.btngroup_ResampleMethod.addButton(self.rbtnResample2)
        self.btngroup_ResampleMethod.addButton(self.rbtnResample3)
        self.btngroup_ResampleMethod.addButton(self.rbtnResample4)

        lay10 = QHBoxLayout()
        lay10.addWidget(self.labelResampleMethod)
        lay10.addWidget(self.rbtnResample0)
        lay10.addWidget(self.rbtnResample1)
        lay10.addWidget(self.rbtnResample2)
        lay10.addWidget(self.rbtnResample3)
        lay10.addWidget(self.rbtnResample4)

        self.labelDefaultSettings = QLabel('Default registration?\t')
        #        self.labelPrefix.setToolTip(setToolTips.LabelPrefixBias())
        self.btngroup_DefaultSettingsRegistration = QButtonGroup()
        self.rbtnDefaultRegistrationy = QRadioButton('yes')
        self.rbtnDefaultRegistrationn = QRadioButton('no')
        self.rbtnDefaultRegistrationy.toggled.connect(self.onClickedRBTN_DefaultRegistration)
        self.rbtnDefaultRegistrationn.toggled.connect(self.onClickedRBTN_DefaultRegistration)

        self.btngroup_DefaultSettingsRegistration.addButton(self.rbtnDefaultRegistrationy)
        self.btngroup_DefaultSettingsRegistration.addButton(self.rbtnDefaultRegistrationn)
        lay11 = QHBoxLayout()
        lay11.addWidget(self.labelDefaultSettings)
        lay11.addWidget(self.rbtnDefaultRegistrationy)
        lay11.addWidget(self.rbtnDefaultRegistrationn)
        lay11.addStretch()

        self.labelCustomRegistration = QLabel('Edit registration settings?\t')
        self.PushButtonViewRegistration = QPushButton("Edit cmdline")
        self.PushButtonViewRegistration.setDisabled(True)
        self.PushButtonViewRegistration.clicked.connect(self.viewRegistrationCmdLine)

        lay12 = QHBoxLayout()
        lay12.addWidget(self.labelCustomRegistration)
        lay12.addWidget(self.PushButtonViewRegistration)
        lay12.addStretch()

        self.settings_list2.addLayout(lay8)
        self.settings_list2.addStretch(1)
        self.settings_list2.addLayout(lay9)
        self.settings_list2.addLayout(lay10)
        self.settings_list2.addStretch(1)
        self.settings_list2.addLayout(lay11)
        self.settings_list2.addLayout(lay12)
        self.settings_list2.addStretch(10)

        # Merge all upper boxes
        self.layout_upper = QHBoxLayout()
        self.layout_upper.addWidget(self.optionboxBias)
        self.layout_upper.addWidget(self.optionboxRegistration)

        # ====================    Create Content for Buttons at the Bottom      ====================
        layout_bottom = QHBoxLayout()
        self.buttonsave = QPushButton('Save settings \nand return')
        self.buttonsave.clicked.connect(self.close)
        self.buttondefault = QPushButton('Load Default \nsettings')
        self.buttondefault.clicked.connect(self.load_default_settings)

        layout_bottom.addStretch(1)
        layout_bottom.addWidget(self.buttonsave)
        layout_bottom.addWidget(self.buttondefault)

        # ====================    Set Content of box and buttoms to General Layout     =======================
        self.layout_tot.addLayout(self.layout_upper)
        self.layout_tot.addLayout(layout_bottom)
        self.get_settings_from_config()

    def get_settings_from_config(self):
        """function which enters the settings according to some cfg variable which is loaded"""

        if self.cfg == "":
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No default settings found, please double check the folder content. Continuing "
                           "with same settings. ")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Close)
            msgBox.exec()
        else:
            # Left side, i.e. N4BiasCorrection
            self.lineEditPrefix.setText(self.cfg["preprocess"]["ANTsN4"]["prefix"])

            if self.cfg["preprocess"]["ANTsN4"]["denoise"] == 'yes':
                self.rbtnDenoisey.setChecked(True)
            else:
                self.rbtnDenoisen.setChecked(True)

            if self.cfg["preprocess"]["registration"]["resample_method"] == 1:
                self.rbtnResample1.setChecked(True)
            elif self.cfg["preprocess"]["registration"]["resample_method"] == 2:
                self.rbtnResample2.setChecked(True)
            elif self.cfg["preprocess"]["registration"]["resample_method"] == 3:
                self.rbtnResample3.setChecked(True)
            elif self.cfg["preprocess"]["registration"]["resample_method"] == 4:
                self.rbtnResample4.setChecked(True)
            else:
                self.rbtnResample0.setChecked(True)

            if self.cfg["preprocess"]["registration"]["default_registration"] == 'yes':
                self.rbtnDefaultRegistrationy.setChecked(True)

            self.lineEditShrink.setText(str(self.cfg["preprocess"]["ANTsN4"]["shrink-factor"]))
            self.lineEditBSplineDist.setText(str(self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"]))
            self.lineEditConv1.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][0]))
            self.lineEditConv2.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][1]))
            self.lineEditConv3 .setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][2]))
            self.lineEditConv4.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][3]))
            self.lineEditTolerance.setText(str(self.cfg["preprocess"]["ANTsN4"]["threshold"]))
            self.lineEditDiffPrefix.setText(str(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]))

            # Right side, i.e. Registration

            self.lineEditPrefixRegistration.setText(self.cfg["preprocess"]["registration"]["prefix"])
            self.lineResampleSpacing.setText(str(self.cfg["preprocess"]["registration"]["resample_spacing"]))


    def closeEvent(self, event):
        """saves the settings found here as a yaml file which may be loaded the next time as the configuration used"""

        self.cfg["preprocess"]["ANTsN4"]["prefix"] = self.lineEditPrefix.text()
        self.cfg["preprocess"]["ANTsN4"]["shrink-factor"] = int(self.lineEditShrink.text())
        self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"] = int(self.lineEditBSplineDist.text())
        self.cfg["preprocess"]["ANTsN4"]["convergence"][0] = int(self.lineEditConv1.text())
        self.cfg["preprocess"]["ANTsN4"]["convergence"][1] = int(self.lineEditConv2.text())
        self.cfg["preprocess"]["ANTsN4"]["convergence"][2] = int(self.lineEditConv3.text())
        self.cfg["preprocess"]["ANTsN4"]["convergence"][3] = int(self.lineEditConv4.text())
        self.cfg["preprocess"]["ANTsN4"]["threshold"] = float(self.lineEditTolerance.text())
        self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] = self.lineEditDiffPrefix.text()

        self.cfg["preprocess"]["registration"]["prefix"] = self.lineEditPrefixRegistration.text()
        self.cfg["preprocess"]["registration"]["resample_spacing"] = self.lineResampleSpacing.text()

        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)
        event.accept()

    def load_default_settings(self):
        """loads the default settings as per the file in the private folder; for that a confirmation is necessary"""

        ret = QMessageBox.question(self, 'MessageBox', "Do you really want to restore default settings for ANTs?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            with open(os.path.join(ROOTDIR, 'private/') + 'config_imagingTBdef.yaml', 'r') as cfg:
                cfg_temp = yaml.safe_load(cfg)
                self.cfg["preprocess"]["ANTsN4"] = cfg_temp["preprocess"]["ANTsN4"]
                self.cfg["preprocess"]["registration"] = cfg_temp["preprocess"]["registration"]
        self.get_settings_from_config()

    @QtCore.pyqtSlot()
    def viewRegistrationCmdLine(self):
        """opens the CmdLine to write custom registration routines. For the correct arguments please have a look at the
        homepage from ANTs documentation (e.g.https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call).
        A n example of how to implement something similar can be found in the [private] directory """
        from subprocess import Popen

        if sys.platform == 'linux':
            Popen(["gedit", os.path.join(ROOTDIR, 'utils', 'cmdline_ANTsRegistration.txt')],
                  stdin=open(os.devnull, 'r'))
        elif sys.platform == 'macos':
            print('Not yet implemented!!')

    # In the next lines, actions are defined when rButtons are pressed; Principally, button is checked and cfg updated
    @QtCore.pyqtSlot()
    def onClickedRBTN_shrink(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["ANTsN4"]["shrink"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_Denoise(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["ANTsN4"]["denoise"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_ResampleMethod(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["registration"]["resample_method"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_DefaultRegistration(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["registration"]["default_registration"] = self.sender().text()

        if self.sender().text() == 'no':
            self.PushButtonViewRegistration.setEnabled(True)
        else:
            self.PushButtonViewRegistration.setEnabled(False)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GuiSettingsNiftiAnts()
    sys.exit(app.exec_())
