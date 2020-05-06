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


class GuiSettingsNiftiAnts(QWidget):
    """ Helper GUI, which enables to change the settings of the different steps with the ANts Toolbox to analyse NIFTI
      data"""

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent=None)
        # Load configuration files and general settings
        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.cfg = HF.LittleHelpers.load_config(rootdir)

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

        self.labelShrink = QLabel('Shrink factor?\t\t')
        self.labelShrink.setToolTip(setToolTips.LabelShrink())
        self.lineEditShrink = QLineEdit()
        regex = QtCore.QRegExp('^[0-4]\d{0}$')
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditShrink.setValidator(validator1)

        lay2 = QHBoxLayout()
        lay2.addWidget(self.labelShrink)
        lay2.addWidget(self.lineEditShrink)
        lay2.addStretch()

        # TODO double check the Label and the ToolTip to ensure accuracy
        self.labelBSplineDist = QLabel('Distance BSplines?\t')
        self.labelBSplineDist.setToolTip(
            HF.LittleHelpers.split_lines("???To lessen computation time, the input image can be resampled. "
                                         "The shrink factor,specified as single integer, describes this "
                                         "resampling. Shrink factors <= 4 are commonly used."))
        self.lineEditBSplineDist = QLineEdit()
        regex = QtCore.QRegExp('^[0-9]\d{2}$')
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditBSplineDist.setValidator(validator1)

        lay3 = QHBoxLayout()
        lay3.addWidget(self.labelBSplineDist)
        lay3.addWidget(self.lineEditBSplineDist)
        lay3.addStretch()

        # TODO: Change labelConvergence
        width = 46
        self.labelConv = QLabel('Convergence?\t\t')
        self.labelConv.setToolTip("???")
        self.lineEditConv1 = QLineEdit()
        self.lineEditConv1.setFixedWidth(width)
        self.lineEditConv2 = QLineEdit()
        self.lineEditConv2.setFixedWidth(width)
        self.lineEditConv3 = QLineEdit()
        self.lineEditConv3.setFixedWidth(width)
        self.lineEditConv4 = QLineEdit()
        self.lineEditConv4.setFixedWidth(width)

        lay4 = QHBoxLayout()
        lay4.addWidget(self.labelConv)
        lay4.addWidget(self.lineEditConv1)
        lay4.addWidget(self.lineEditConv2)
        lay4.addWidget(self.lineEditConv3)
        lay4.addWidget(self.lineEditConv4)
        lay4.addStretch()

        self.labelTolerance = QLabel('Tolerance?\t\t')
        self.labelTolerance.setToolTip("")
        self.lineEditTolerance = QLineEdit()
        # TODO: set validator for values < 0

        lay5 = QHBoxLayout()
        lay5.addWidget(self.labelTolerance)
        lay5.addWidget(self.lineEditTolerance)
        lay5.addStretch()

        self.labelDiffPrefix = QLabel('Prefix for DTI data?\t')
        self.labelDiffPrefix.setToolTip("")
        self.lineEditDiffPrefix = QLineEdit()

        lay6 = QHBoxLayout()
        lay6.addWidget(self.labelDiffPrefix)
        lay6.addWidget(self.lineEditDiffPrefix)
        lay6.addStretch()

        self.settings_list1.addLayout(lay1)
        self.settings_list1.addLayout(lay2)
        self.settings_list1.addLayout(lay3)
        self.settings_list1.addLayout(lay4)
        self.settings_list1.addLayout(lay5)
        self.settings_list1.addLayout(lay6)
        self.settings_list1.addStretch()

        # ==============================    Create Content for Right Upper Box   ==============================
        self.optionboxRegistration = QGroupBox('ImageRegistration')
        self.settings_list2 = QVBoxLayout(self.optionboxRegistration)
        self.settings_list2.addLayout(lay1)

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
            self.lineEditShrink.setText(str(self.cfg["preprocess"]["ANTsN4"]["shrink-factor"]))
            self.lineEditBSplineDist.setText(str(self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"]))
            self.lineEditConv1.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][0]))
            self.lineEditConv2.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][1]))
            self.lineEditConv3.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][2]))
            self.lineEditConv4.setText(str(self.cfg["preprocess"]["ANTsN4"]["convergence"][3]))
            self.lineEditTolerance.setText(str(self.cfg["preprocess"]["ANTsN4"]["threshold"]))
            self.lineEditDiffPrefix.setText(str(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]))

            # Right side, i.e. Registration

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

        HF.LittleHelpers.save_config(self.cfg["folders"]["rootdir"], self.cfg)
        event.accept()

    def load_default_settings(self):
        """loads the default settings as per the file in the private folder; for that a confirmation is necessary"""

        ret = QMessageBox.question(self, 'MessageBox', "Do you really want to restore default settings for ANTs?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            with open(os.path.join(rootdir, 'private/') + 'config_imagingTBdef.yaml', 'r') as cfg:
                cfg_temp = yaml.safe_load(cfg)
                self.cfg["preprocess"]["ANTsN4"] = cfg_temp["preprocess"]["ANTsN4"]
        self.get_settings_from_config()

    # In the next lines, actions are defined when rbtns are pressed; Principally, button is checked and cfg is updated
    @QtCore.pyqtSlot()
    def onClickedRBTN_shrink(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["ANTsN4"]["shrink"] = self.sender().text()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GuiSettingsNiftiAnts()
    sys.exit(app.exec_())


# Functions to use in ANTs

# only perform on MRI data and no DTI (-> in this case throw error not implemented yet)
# check values being > 0

# N4BiasFieldCorrection(image dimensionality = 3, input-image = subj1 ....nii, output bc_....nii, shrink-factor = 4,
# bsline-fitting = [200], convergence 50x50x50x50, 0.000001)

# Debug: visualise results by ratio(pre/post) as overlay on original image
