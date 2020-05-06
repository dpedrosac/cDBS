#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import warnings
import sys
import re
import glob
import subprocess
from PyQt5.QtWidgets import QMessageBox


class LittleHelpers:
    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """ copied from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)

        # Print New Line on Complete
        if iteration == total:
            print()

    @staticmethod
    def logging_routine(text, cfg, subject, module, opt, project="MyoDBS"):
        """creates a logfile which makes sure, all steps can be traced back after doing the analyses. For that purpose,
        information from the yaml-file is needed.
        @params:
            text        - Required  : Text to write into file after summarizing the settings/options (Str)
            cfg         - Required  : configuration as per yaml file (Dict)
            module      - Required  : Action that is performed (Str)
            opt         - Optional  : module specific configuration
            project     - Optional  : Name of the project (Str)
        """

        logfile = os.path.join(cfg["folders"]["nifti"], subject, "pipeline_" + subject + '.txt')
        if not os.path.isfile(logfile):
            # this part creates the first part of the logging routine, i.e. something like a header
            outfile = open(logfile, 'w+')
            hdr = "="*130 + "\nAnalysis pipeline for " + project +" project.\n\nThis file summarises the steps taken " \
                            "so that at the end, all preprocessing steps and options can be reproduced\n \nLog-File " \
                            "for: \t\t{}\n \n".format(subject) + "="*130
            outfile.write(LittleHelpers.split_lines(hdr))
            outfile.close()

        with open(logfile, 'a+') as outfile:
            if opt:
                string_opt = ''.join("{:<20s}:{}\n".format(n, v) for n, v in opt.items())
            else:
                string_opt = ''.join("{:<20s}:{}\n".format("None"))

            output = "\nRunning {} for subj {} with the following options:\n" \
                              "\n{}\n".format(module, subject, string_opt) + "-"*130
            outfile.write("\n" + LittleHelpers.split_lines(output))
            outfile.write("\n\n" + text + "\n")

        outfile.close

    @staticmethod
    def split_lines(text):
        lines = text.split('\n')
        regex = re.compile(r'.{1,130}(?:\s+|$)')
        return '\n'.join(s.rstrip() for line in lines for s in regex.findall(line))

    @staticmethod
    def load_config(maindir):
        """loads the configuration saved in the yaml file in order to use or update the content in a separate file"""

        try:
            with open(os.path.join(maindir, 'config_imagingTB.yaml'), 'r') as yfile:
                cfg = yaml.safe_load(yfile)
        except FileNotFoundError:
            warnings.warn("No valid configuration file was found. Using default settings. Please make sure that a file "
                          "named config_imagingTB is in the main folder of the imaging toolbox")
            with open(os.path.join(maindir, 'private') + 'config_imagingTBdef.yaml', 'r') as yfile:
                cfg = yaml.safe_load(yfile)

        return cfg

    @staticmethod
    def save_config(rootdir, cfg):
        """saves the configuration in a yaml file in order to use or update the content in a separate file"""

        with open(os.path.join(rootdir, 'config_imagingTB.yaml'), 'wb') as settings_mod:
            yaml.safe_dump(cfg, settings_mod, default_flow_style=False,
                           explicit_start=True, allow_unicode=True, encoding='utf-8')

    @staticmethod
    def load_imageviewer(path2viewer, imagefolder, suffix=''):
        """loads NIFTI-files to pathviewer"""

        file_names = glob.glob(os.path.join(imagefolder, '*.nii'))

        if not file_names:
            txt = "There are no NIFTI-files in this folder ({}), please double-check".format(imagefolder)
            title = "No NIFTI-files available"
            LittleHelpers.msg_box(txt, title)
            return

        #TODO: implement a way to get single files read on winxx platforms as well.

        if sys.platform == ('win32' or 'win64'):
            cmd = [path2viewer + '/ITK-SNAP.exe -g {0} -o '.format(file_names[0]) + ' '.join(file_names[1:])]
        elif sys.platform == ('linux' or 'linux2'):
            if len(file_names) == 1:
                cmd = ["itksnap", "-g", file_names[0]]
            else:
                cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]
        elif sys.platform == 'macos':
            LittleHelpers.msg_box(text="Could not be tested yet!!!", title="Mac unavailable so far")

        if ('ITK-SNAP' in path2viewer or 'snap' in path2viewer):
            subprocess.call(cmd)
        else:
            LittleHelpers.msg_box(text="Viewers other than ITK-SNAP are not implemented!",
                                  title="Unknown viewer")

    @staticmethod
    def msg_box(text='Unknown text', title='unknown title', flag='Information'):
        """helper intended to provide some sort of message box with a text and a title"""
        msgBox = QMessageBox()
        if flag == 'Information':
            msgBox.setIcon(QMessageBox.Information)
        elif flag == 'Warning':
            msgBox.setIcon(QMessageBox.Warning)
        else:
            msgBox.setIcon(QMessageBox.Critical)

        msgBox.setText(text)
        msgBox.setWindowTitle(title)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

