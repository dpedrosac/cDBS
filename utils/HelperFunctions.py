#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import warnings
import sys
import re
import subprocess
from PyQt5.QtWidgets import QMessageBox
import private.CheckDefaultFolders as LocationCheck

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

            output = "\nRunning {} for {} with the following options:\n" \
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
    def load_imageviewer(path2viewer, file_names, suffix=''):
        """loads selected NIFTI-files in imageviewer"""

        #file_names = glob.glob(os.path.join(imagefolder, '*.nii'))

        if not file_names:
            LittleHelpers.msg_box(txt="The provided list with NIFTI-files is empty, please double-check",
                                  title="No NIFTI-files provided")
            return

        # TODO: implement a way to get single files read on winxx platforms as well.

        if sys.platform == ('win32' or 'win64'):
            cmd = [path2viewer + '/ITK-SNAP.exe -g {0} -o '.format(file_names[0]) + ' '.join(file_names[1:])]
        elif sys.platform == ('linux' or 'linux2'):
            if len(file_names) == 1:
                cmd = ["itksnap", "-g", file_names[0]]
            else:
                cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]
        elif sys.platform == 'macos':
            LittleHelpers.msg_box(text="Could not be tested yet!!!", title="Mac unavailable so far")

        if 'ITK-SNAP' in path2viewer or 'snap' in path2viewer:
            p = subprocess.Popen(cmd, shell=False,
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE)
            stdoutdata, stderrdata = p.communicate()
            flag = p.returncode
            if flag != 0:
                print(stderrdata)

        else:
            LittleHelpers.msg_box(text="Viewers other than ITK-SNAP are not implemented!",
                                  title="Unknown viewer")


# General functions: message box, find (sub-)folders in a directory,
def set_viewer(viewer_opt):
    """Sets the viewer_option in the configuration file"""
    from dependencies import ROOTDIR

    cfg = LittleHelpers.load_config(ROOTDIR)
    if viewer_opt == 'itk-snap': # to-date, only one viewer is available. May be changed in a future
        if not cfg["folders"]["path2itksnap"]:
            cfg["folders"]["path2itksnap"] = LocationCheck.FileLocation.itk_snap_check(ROOTDIR)
            LittleHelpers.save_config(ROOTDIR, cfg)

    return cfg["folders"]["path2itksnap"]


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

def list_folders(inputdir, prefix='subj', files2lookfor='NIFTI'):
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

def list_files_in_folder(inputdir, contains='', suffix='nii', subfolders=True):
    """returns a list of files within a folder (including subfolders"""
    import glob

    if subfolders:
        allfiles_in_folder = [os.path.split(x)[1] for x in glob.glob(os.path.join(inputdir + '/**/*.' + suffix),
                                                                     recursive=True)]
    else:
        allfiles_in_folder = [os.path.split(x)[1] for x in glob.glob(inputdir + '/*.' + suffix)]

    filelist = [file_id for file_id in allfiles_in_folder if any(y in file_id for y in contains)]

    return filelist

def get_filelist_as_tuple(inputdir, subjects):
    """create a list of all available files in a folder and returns this list"""
    import glob

    allfiles = []
    [allfiles.extend(zip(glob.glob(os.path.join(inputdir, x + "/*")), [x] * len(glob.glob(os.path.join(inputdir, x + "/*")))))
     for x in subjects]

    return allfiles

def display_files_in_viewer(inputdir, regex2include, regex2exclude, selected_subjects='', viewer='itk-snap'):
    """Routine intended to provide generic way of displaying a batch of data using the viewer selected"""
    from dependencies import ROOTDIR
    import glob

    cfg = LittleHelpers.load_config(ROOTDIR)

    if not selected_subjects:
        msg_box(text="No folder selected. To proceed, please indicate what folder to process.",
                   title="No subject selected")
        return
    elif len(selected_subjects) > 1:
        msg_box(text="Please select only one folder to avoid loading too many images",
                title="Too many subjects selected")
        return
    else:
        image_folder = os.path.join(cfg["folders"]["nifti"], selected_subjects[0])

    viewer_path = set_viewer(viewer)  # to-date, only one viewer is available. May be changed in a future
    regex_complete = regex2include
    regex2exclude = [''.join('~{}'.format(y)) for y in regex2exclude]
    regex_complete += regex2exclude

    r = re.compile(r"^~.*")
    excluded_sequences = [x[1:] for x in list(filter(r.match, regex_complete))]
    included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

    all_files = glob.glob(image_folder + '/**/*.nii', recursive=True) # returns all available NIFTI-files in the folder including subfolders

    file_IDs = []
    [file_IDs.append(x) for x in all_files for y in included_sequences
     if re.search(r'\w+{}.'.format(y), x, re.IGNORECASE)]

    LittleHelpers.load_imageviewer(viewer_path, sorted(file_IDs))
