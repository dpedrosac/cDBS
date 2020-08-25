#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import yaml
import os
import utils.HelperFunctions as HF
import private.allToolTips as setToolTips
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGroupBox, QVBoxLayout, QHBoxLayout, \
    QPushButton, QButtonGroup, QRadioButton, QLineEdit, QMessageBox, QComboBox
from dependencies import ROOTDIR


class GuiLeadDetection(QWidget):
    """ Helper GUI, which enables to set the different options for the lead detection"""

    def __init__(self, parent=None):
        super(GuiLeadDetection, self).__init__(parent=None)

        # Load configuration files and general settings
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)

        # General appearance of the GUI
        self.setFixedSize(800, 600)
        self.setWindowTitle('Settings for Lead Detection using the different packages')
        self.show()

        # Create general layout
        self.layout_tot = QVBoxLayout(self)

        # ==============================    Create Content for Left Upper Box   ==============================
        self.optionboxPaCER = QGroupBox('PaCER')
        self.settings_list1 = QVBoxLayout(self.optionboxPaCER)

        self.labelMetalThreshold = QLabel('Metal threshold?\t\t')
        self.labelMetalThreshold.setToolTip(setToolTips.PaCER_MetalThreshold())
        self.lineEditMetalThreshold = QLineEdit()

        lay1 = QHBoxLayout()
        lay1.addWidget(self.labelMetalThreshold)
        lay1.addWidget(self.lineEditMetalThreshold)
        lay1.addStretch()

        self.labelSNRThreshold = QLabel('SNR threshold?\t\t')
        #self.labelSNRThreshold.setToolTip(setToolTips.PaCER_MetalThreshold())
        self.lineEditSNRThreshold = QLineEdit()

        lay2 = QHBoxLayout()
        lay2.addWidget(self.labelSNRThreshold)
        lay2.addWidget(self.lineEditSNRThreshold)
        lay2.addStretch()

        self.labelLambda = QLabel('Lambda?\t\t\t')
        self.labelLambda.setToolTip(setToolTips.PaCER_Lambda())
        self.lineEditLambda = QLineEdit()
        regex = QtCore.QRegExp('^[0-9]\d{1}$')
        validator1 = QtGui.QRegExpValidator(regex)
        self.lineEditLambda.setValidator(validator1)

        lay3 = QHBoxLayout()
        lay3.addWidget(self.labelLambda)
        lay3.addWidget(self.lineEditLambda)
        lay3.addStretch()

        self.labelProbMask = QLabel('Use probabilisitic mask?\t')
        self.labelProbMask.setToolTip(setToolTips.ProbabilisticMask())
        self.btngroup_ProbMask = QButtonGroup()
        self.rbtnProbMasky = QRadioButton('yes')
        self.rbtnProbMaskn = QRadioButton('no')
        self.rbtnProbMasky.toggled.connect(self.onClickedRBTN_ProbMask)
        self.rbtnProbMaskn.toggled.connect(self.onClickedRBTN_ProbMask)

        self.btngroup_ProbMask.addButton(self.rbtnProbMasky)
        self.btngroup_ProbMask.addButton(self.rbtnProbMaskn)
        lay4 = QHBoxLayout()
        lay4.addWidget(self.labelProbMask)
        lay4.addWidget(self.rbtnProbMasky)
        lay4.addWidget(self.rbtnProbMaskn)
        lay4.addStretch()

        self.labelDetectionMethod = QLabel('Detection method?\t')
        self.lineDetectionMethod = QComboBox()
        allowable_methods = [
            "peak",
            "peakWaveCenter",
            "contactAreaCenter"
        ]
        [self.lineDetectionMethod.addItem(x) for x in allowable_methods]
        idx_method = self.lineDetectionMethod.findText(self.cfg["lead_detection"]["PaCER"]["detection_method"],
                                                       QtCore.Qt.MatchFixedString)

        if idx_method >= 0:
            self.lineDetectionMethod.setCurrentIndex(idx_method)
        self.lineDetectionMethod.currentTextChanged.connect(self.comboChangedDetectionMethod)
        self.lineDetectionMethod.setDisabled(False)

        lay5 = QHBoxLayout()
        lay5.addWidget(self.labelDetectionMethod)
        lay5.addWidget(self.lineDetectionMethod)
        lay5.addStretch()

        self.labelLeadType = QLabel('Implanted lead?\t\t')
        self.lineLeadType = QComboBox()
        allowable_methods = [
            "MDT-3387",
            "MDT-3389",
            "BSc-2201 (non-direct.)",
            "BSc-2202 (direct.)"
        ]
        [self.lineLeadType.addItem(x) for x in allowable_methods]
        idx_method = self.lineLeadType.findText(self.cfg["lead_detection"]["PaCER"]["lead_type"],
                                                QtCore.Qt.MatchFixedString)
        if idx_method >= 0:
            self.lineLeadType.setCurrentIndex(idx_method)
        self.lineLeadType.currentTextChanged.connect(self.comboChangedLeadType)
        self.lineLeadType.setDisabled(False)

        lay6 = QHBoxLayout()
        lay6.addWidget(self.labelLeadType)
        lay6.addWidget(self.lineLeadType)
        lay6.addStretch()

        self.labelTransformMatrix = QLabel('Use transformation matrix?\t')
        self.labelProbMask.setToolTip(setToolTips.ProbabilisticMask())
        self.btngroup_TransformMatrx = QButtonGroup()
        self.rbtnTransformMatrixy = QRadioButton('yes')
        self.rbtnTransformMatrixn = QRadioButton('no')
        self.rbtnProbMasky.toggled.connect(self.onClickedRBTN_TransformationMatrix)
        self.rbtnProbMaskn.toggled.connect(self.onClickedRBTN_TransformationMatrix)

        self.btngroup_TransformMatrx.addButton(self.rbtnTransformMatrixy)
        self.btngroup_TransformMatrx.addButton(self.rbtnTransformMatrixn)
        lay7 = QHBoxLayout()
        lay7.addWidget(self.labelTransformMatrix)
        lay7.addWidget(self.rbtnTransformMatrixy)
        lay7.addWidget(self.rbtnTransformMatrixn)
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
        # self.optionboxRegistration = QGroupBox('Other')
        # self.settings_list2 = QVBoxLayout(self.optionboxMarALAGO)
        # self.settings_list2.addLayout(lay1)

        # self.labelPrefixRegistration = QLabel('Registration prefix?\t')
        # self.labelPrefixRegistration.setToolTip(setToolTips.LabelPrefixBias())
        # self.lineEditPrefixRegistration = QLineEdit()

        # lay8 = QHBoxLayout()
        # lay8.addWidget(self.labelPrefixRegistration)
        # lay8.addWidget(self.lineEditPrefixRegistration)
        # lay8.addStretch()

        # self.settings_list2.addLayout(lay8)
        # self.settings_list2.addStretch(1)

        # Merge all upper boxes
        self.layout_upper = QHBoxLayout()
        self.layout_upper.addWidget(self.optionboxPaCER)
        # self.layout_upper.addWidget(self.optionboxMarALAGO)

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
            # Left side, i.e. PaCER
            self.lineEditMetalThreshold.setText(self.cfg["lead_detection"]["PaCER"]["metal_threshold"])
            self.lineEditSNRThreshold.setText(self.cfg["lead_detection"]["PaCER"]["snr_threshold"])
            self.lineEditLambda.setText(str(self.cfg["lead_detection"]["PaCER"]["lambda"]))

            if self.cfg["lead_detection"]["PaCER"]["probabilistic_mask"] == 'yes':
                self.rbtnProbMasky.setChecked(True)
            else:
                self.rbtnProbMaskn.setChecked(True)

            if self.cfg["lead_detection"]["PaCER"]["transformation_matrix"] == 'yes':
                self.rbtnTransformMatrixy.setChecked(True)
            else:
                self.rbtnTransformMatrixn.setChecked(True)

            # Right side, i.e. Registration
            # self.lineEditPrefixRegistration.setText(self.cfg["preprocess"]["registration"]["prefix"])

    def closeEvent(self, event):
        """saves the settings found here as a yaml file which may be loaded the next time as the configuration used"""

        self.cfg["lead_detection"]["PaCER"]["metal_threshold"] = self.lineEditMetalThreshold.text()
        self.cfg["lead_detection"]["PaCER"]["snr_threshold"] = self.lineEditSNRThreshold.text()
        self.cfg["lead_detection"]["PaCER"]["lambda"] = int(self.lineEditLambda.text())

        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)
        event.accept()

    def load_default_settings(self):
        """loads the default settings as per the file in the private folder; for that a confirmation is necessary"""

        ret = QMessageBox.question(self, 'MessageBox', "Do you really want to restore default settings for Lead "
                                                       "detection?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            with open(os.path.join(ROOTDIR, 'private/') + 'config_imagingTBdef.yaml', 'r') as cfg:
                cfg_temp = yaml.safe_load(cfg)
                self.cfg["lead_detection"]["PaCER"] = cfg_temp["lead_detection"]["PaCER"]
                # self.cfg["lead_detection"]["MarALAGO"] = cfg_temp["lead_detection"]["MarALAGO"]
        self.get_settings_from_config()

    # In the next lines, actions are defined when rButtons are pressed; Principally, button is checked and cfg updated
    @QtCore.pyqtSlot()
    def onClickedRBTN_ProbMask(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["lead_detection"]["PaCER"]["probabilistic_mask"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_TransformationMatrix(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["lead_detection"]["PaCER"]["transformation_matrix"] = self.sender().text()

    @QtCore.pyqtSlot()
    def onClickedRBTN_DefaultRegistration(self):
        radioBtn = self.sender()
        radioBtn.isChecked()
        self.cfg["preprocess"]["registration"]["default_registration"] = self.sender().text()

    @QtCore.pyqtSlot()
    def comboChangedDetectionMethod(self):
        self.cfg["lead_detection"]["PaCER"]["detection_method"] = self.lineDetectionMethod.currentText()
        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)

    @QtCore.pyqtSlot()
    def comboChangedLeadType(self):
        self.cfg["lead_detection"]["PaCER"]["lead_type"] = self.lineLeadType.currentText()
        HF.LittleHelpers.save_config(ROOTDIR, self.cfg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = GuiLeadDetection()
    sys.exit(app.exec_())
