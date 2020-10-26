#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import re
import subprocess
import sys
import warnings
from itertools import groupby
from operator import itemgetter

import numpy as np
import yaml
from PyQt5.QtWidgets import QMessageBox

import private.CheckDefaultFolders as LocationCheck
from dependencies import ROOTDIR, GITHUB


class LittleHelpers:
    def __init__(self, _debug=False):
        self.debug = _debug

#     @staticmethod
#     def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
#         """ copied from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#         Call in a loop to create terminal progress bar
#         @params:
#             iteration   - Required  : current iteration (Int)
#             total       - Required  : total iterations (Int)
#             prefix      - Optional  : prefix string (Str)
#             suffix      - Optional  : suffix string (Str)
#             decimals    - Optional  : positive number of decimals in percent complete (Int)
#             length      - Optional  : character length of bar (Int)
#             fill        - Optional  : bar fill character (Str)
#             printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
#         """
#         percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#         filledLength = int(length * iteration // total)
#         bar = fill * filledLength + '-' * (length - filledLength)
#         print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
#
#         # Print New Line on Complete
#         if iteration == total:
#             print()
#
#     @staticmethod
#     def logging_routine(text, cfg, subject, module, opt, project="MyoDBS"):
#         """creates a logfile which makes sure, all steps can be traced back after doing the analyses. For that purpose,
#         information from the yaml-file is needed.
#         @params:
#             text        - Required  : Text to write into file after summarizing the settings/options (Str)
#             cfg         - Required  : configuration as per yaml file (Dict)
#             module      - Required  : Action that is performed (Str)
#             opt         - Optional  : module specific configuration
#             project     - Optional  : Name of the project (Str)
#         """
#
#         logfile = os.path.join(cfg["folders"]["nifti"], subject, "pipeline_" + subject + '.txt')
#         if not os.path.isfile(logfile):
#             # this part creates the first part of the logging routine, i.e. something like a header
#             outfile = open(logfile, 'w+')
#             hdr = "=" * 110 + "\nAnalysis pipeline for " + project + " project.\n\nThis file summarises the steps taken" \
#                                                                      " so that at the end, all preprocessing steps and " \
#                                                                      "options can be reproduced\n \nLog-File " \
#                                                                      "for: \t\t{}\n \n".format(subject) + "=" * 110
#             outfile.write(LittleHelpers.split_lines(hdr))
#             outfile.close()
#
#         with open(logfile, 'a+') as outfile:
#             if opt:
#                 string_opt = ''.join("{:<20s}:{}\n".format(n, v) for n, v in opt.items())
#             else:
#                 string_opt = ''.join("{:<20s}:{}\n".format('None', 'None'))
#
#             output = "\nRunning {} for {} with the following options:\n" \
#                      "\n{}\n".format(module, subject, string_opt) + "-" * 110
#             outfile.write("\n" + LittleHelpers.split_lines(output))
#             outfile.write("\n\n" + text + "\n")
#             outfile.write("\n" + "-" * 110)
#
#         outfile.close
#
#     @staticmethod
#     def split_lines(text):
#         lines = text.split('\n')
#         regex = re.compile(r'.{1,130}(?:\s+|$)')
#         return '\n'.join(s.rstrip() for line in lines for s in regex.findall(line))
#
#     @staticmethod
#     def load_config(maindir):
#         """loads the configuration saved in the yaml file in order to use or update the content in a separate file"""
#
#         try:
#             with open(os.path.join(maindir, 'config_imagingTB.yaml'), 'r') as yfile:
#                 cfg = yaml.safe_load(yfile)
#         except FileNotFoundError:
#             warnings.warn("No valid configuration file was found. Using default settings. Please make sure that a file "
#                           "named config_imagingTB is in the main folder of the imaging toolbox")
#             with open(os.path.join(maindir, 'private') + 'config_imagingTBdef.yaml', 'r') as yfile:
#                 cfg = yaml.safe_load(yfile)
#
#         return cfg
#
#     @staticmethod
#     def save_config(rootdir, cfg):
#         """saves the configuration in a yaml file in order to use or update the content in a separate file"""
#
#         with open(os.path.join(rootdir, 'config_imagingTB.yaml'), 'wb') as settings_mod:
#             yaml.safe_dump(cfg, settings_mod, default_flow_style=False,
#                            explicit_start=True, allow_unicode=True, encoding='utf-8')
#
#     @staticmethod
#     def load_imageviewer(path2viewer, file_names, suffix=''):
#         """loads selected NIFTI-files in imageviewer"""
#
#         # file_names = glob.glob(os.path.join(imagefolder, '*.nii'))
#
#         if not file_names:
#             Output.msg_box(txt="The provided list with NIFTI-files is empty, please double-check",
#                            title="No NIFTI-files provided")
#             return
#
#         if sys.platform == ('win32' or 'win64'):
#             cmd = [path2viewer + '/ITK-SNAP.exe -g {0} -o '.format(file_names[0]) + ' '.join(file_names[1:])]
#         elif sys.platform == ('linux' or 'linux2'):
#             if len(file_names) == 1:
#                 cmd = ["itksnap", "-g", file_names[0]]
#             else:
#                 cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]
#         elif sys.platform == 'macos' or sys.platform == 'darwin':
#             # LittleHelpers.msg_box(text="Could not be tested yet!!!", title="Mac unavailable so far")
#             if len(file_names) == 1:
#                 cmd = ["itksnap", "-g", file_names[0]]
#             else:
#                 cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]
#
#             # TODO: add a setup file which ensures the right structure is given w/ data/, ext/, etc.
#             # TODO: add sudo /Applications/ITK-SNAP.app/Contents/bin/install_cmdl.sh to the setup
#             # TODO: change ITK-SNAP so that it does not freeze the entire script
#             # TODO: remove other viewers apart from itksnap and options for win systems
#
#         if 'ITK-SNAP' in path2viewer or 'snap' in path2viewer:
#             p = subprocess.Popen(cmd, shell=False,
#                                  stdin=subprocess.PIPE,
#                                  stdout=subprocess.DEVNULL,
#                                  stderr=subprocess.PIPE)
#             stdoutdata, stderrdata = p.communicate()
#             flag = p.returncode
#             if flag != 0:
#                 print(stderrdata)
#
#         else:
#             Output.msg_box(text="Viewers other than ITK-SNAP are not implemented!",
#                            title="Unknown viewer")
#
#     @staticmethod
#     def create_folder(foldername):
#         """creates folder if not existent"""
#
#         if not os.path.isdir(foldername):
#             os.mkdir(foldername)
#
#
# # General functions: message box, find (sub-)folders in a directory,
# def set_viewer(viewer_opt):
#     """Sets the viewer_option in the configuration file"""
#     from dependencies import ROOTDIR
#
#     cfg = LittleHelpers.load_config(ROOTDIR)
#     if viewer_opt == 'itk-snap':  # to-date, only one viewer is available. May be changed in a future
#         if not cfg["folders"]["path2itksnap"]:
#             cfg["folders"]["path2itksnap"] = LocationCheck.FileLocation.itk_snap_check(ROOTDIR)
#             LittleHelpers.save_config(ROOTDIR, cfg)
#
#     return cfg["folders"]["path2itksnap"]
#
#
# def msg_box(text='Unknown text', title='unknown title', flag='Information'):
#     """helper intended to provide some sort of message box with a text and a title"""
#     msgBox = QMessageBox()
#     if flag == 'Information':
#         msgBox.setIcon(QMessageBox.Information)
#     elif flag == 'Warning':
#         msgBox.setIcon(QMessageBox.Warning)
#     else:
#         msgBox.setIcon(QMessageBox.Critical)
#
#     msgBox.setText(text)
#     msgBox.setWindowTitle(title)
#     msgBox.setStandardButtons(QMessageBox.Ok)
#     msgBox.exec()
#
#
# def list_folders(inputdir, prefix='subj', files2lookfor='NIFTI'):
#     """takes folder and lists all available subjects in this folder according to some filter given as [prefix]"""
#
#     list_all = [name for name in os.listdir(inputdir)
#                 if (os.path.isdir(os.path.join(inputdir, name)) and prefix in name)]
#
#     if list_all == '':
#         list_subj = 'No available subjects, please make sure {}-files are present and correct ' \
#                     '"prefix" is given'.format(files2lookfor)
#     else:
#         list_subj = set(list_all)
#
#     return list_subj
#
#
# def list_files_in_folder(inputdir, contains='', suffix='nii', entire_path=False, subfolders=True):
#     """returns a list of files within a folder (including subfolders"""
#
#     if subfolders:
#         allfiles_in_folder = glob.glob(os.path.join(inputdir + '/**/*.' + suffix), recursive=True)
#     else:
#         allfiles_in_folder = glob.glob(inputdir + '/*.' + suffix)
#
#     if not contains:
#         filelist = [file_id for file_id in allfiles_in_folder]
#     else:
#         filelist = [file_id for file_id in allfiles_in_folder if any(y in file_id for y in contains)]
#
#     if not entire_path:
#         filelist = [os.path.split(x)[1] for x in filelist]
#
#     return filelist
#
#
# def get_filelist_as_tuple(inputdir, subjects):
#     """create a list of all available files in a folder and returns this list"""
#     import glob
#
#     allfiles = []
#     [allfiles.extend(
#         zip(glob.glob(os.path.join(inputdir, x + "/*")), [x] * len(glob.glob(os.path.join(inputdir, x + "/*")))))
#         for x in subjects]
#
#     return allfiles
#
#
# def display_files_in_viewer(inputdir, regex2include, regex2exclude, selected_subjects='', viewer='itk-snap'):
#     """Routine intended to provide generic way of displaying a batch of data using the viewer selected"""
#     import glob
#
#     cfg = LittleHelpers.load_config(ROOTDIR)
#
#     if not selected_subjects:
#         msg_box(text="No folder selected. To proceed, please indicate what folder to process.",
#                 title="No subject selected")
#         return
#     elif len(selected_subjects) > 1:
#         msg_box(text="Please select only one folder to avoid loading too many images",
#                 title="Too many subjects selected")
#         return
#     else:
#         image_folder = os.path.join(cfg["folders"]["nifti"], selected_subjects[0])
#
#     viewer_path = set_viewer(viewer)  # to-date, only one viewer is available. May be changed in a future
#     regex_complete = regex2include
#     regex2exclude = [''.join('~{}'.format(y)) for y in regex2exclude]
#     regex_complete += regex2exclude
#
#     r = re.compile(r"^~.*")
#     excluded_sequences = [x[1:] for x in list(filter(r.match, regex_complete))]
#     included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]
#
#     all_files = glob.glob(image_folder + '/**/*.nii',
#                           recursive=True)  # returns all available NIFTI-files in the folder including subfolders
#
#     file_IDs = []
#     [file_IDs.append(x) for x in all_files for y in included_sequences
#      if re.search(r'\w+{}.'.format(y), x, re.IGNORECASE)]
#
#     LittleHelpers.load_imageviewer(viewer_path, sorted(file_IDs))
#
#
# def resampleANTsImaging(mm_spacing, ANTsImageObject, file_id, method):
#     """Function aiming at rescaling imaging to a resolution specified somewhere, e.g. in the cfg-file"""
#     import math, ants
#
#     resolution = [mm_spacing] * 3
#     if not all([math.isclose(float(mm_spacing), x, abs_tol=10 ** -2) for x in ANTsImageObject.spacing]):
#         print('Image spacing {:.4f}x{:.4f}x{:.4f} unequal to specified value ({}mm). '
#               '\n\tRescaling {}'.format(ANTsImageObject.spacing[0], ANTsImageObject.spacing[1],
#                                         ANTsImageObject.spacing[2], mm_spacing, file_id))
#
#         if len(ANTsImageObject.spacing) == 4:
#             resolution.append(ANTsImageObject.spacing[-1])
#         elif len(ANTsImageObject.spacing) > 4:
#             msg_box(text='Sequences of >4 dimensions are not possible.', title='Too many dimensions')
#
#         resampled_image = ants.resample_image(ANTsImageObject, (resolution), use_voxels=False, interp_type=method)
#         ants.image_write(resampled_image, filename=file_id)
#     else:
#         print('Image spacing for sequence: {} is {:.4f}x{:.4f}x{:.4f} as specified in options, '
#               'proceeding'.format(file_id, ANTsImageObject.spacing[0], ANTsImageObject.spacing[1],
#                                   ANTsImageObject.spacing[2]))
#         resampled_image = ANTsImageObject
#
#     return resampled_image
#
#
# def inner_join(a, b):
#     """adapted from: https://stackoverflow.com/questions/31887447/how-do-i-merge-two-lists-of-tuples-based-on-a-key"""
#
#     L = a + b
#     L.sort(key=itemgetter(1))  # sort by the first column
#     for _, group in groupby(L, itemgetter(1)):
#         row_a, row_b = next(group), next(group, None)
#         if row_b is not None:  # join
#             yield row_b[0:1] + row_a  # cut 1st column from 2nd row
#
#
# def sphere(diameter):
#     """function defining binary matrix which represents a 3D sphere which may be used as structuring element"""
#
#     struct = np.zeros((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
#     x, y, z = np.indices((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
#     mask = (x - diameter) ** 2 + (y - diameter) ** 2 + (z - diameter) ** 2 <= diameter ** 2
#     struct[mask] = 1
#
#     return struct.astype(np.bool)
#
#
# def accumarray(a, accmap):
#     """ from https://stackoverflow.com/questions/16856470/is-there-a-matlab-accumarray-equivalent-in-numpy"""
#
#     ordered_indices = np.argsort(accmap)
#     ordered_accmap = accmap[ordered_indices]
#     _, sum_indices = np.unique(ordered_accmap, return_index=True)
#     cumulative_sum = np.cumsum(a[ordered_indices])[sum_indices - 1]
#
#     result = np.empty(len(sum_indices), dtype=a.dtype)
#     result[:-1] = cumulative_sum[1:]
#     result[-1] = cumulative_sum[0]
#     result[1:] = result[1:] - cumulative_sum[1:]
#
#     return result
#

class Output:
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
            hdr = "=" * 110 + "\nAnalysis pipeline for " + project + " project.\n\nThis file summarises the steps taken " \
                                                                     "so that at the end, all preprocessing steps and options can be reproduced\n \nLog-File " \
                                                                     "for: \t\t{}\n \n".format(subject) + "=" * 110
            outfile.write(Output.split_lines(hdr))
            outfile.close()

        with open(logfile, 'a+') as outfile:
            if opt:
                string_opt = ''.join("{:<20s}:{}\n".format(n, v) for n, v in opt.items())
            else:
                string_opt = ''.join("{:<20s}:{}\n".format("None"))

            output = "\nRunning {} for {} with the following options:\n" \
                     "\n{}\n".format(module, subject, string_opt) + "-" * 110
            outfile.write("\n" + Output.split_lines(output))
            outfile.write("\n\n" + text + "\n")
            outfile.write("\n" + "-" * 110)

        outfile.close

    @staticmethod
    def split_lines(text):
        lines = text.split('\n')
        regex = re.compile(r'.{1,130}(?:\s+|$)')
        return '\n'.join(s.rstrip() for line in lines for s in regex.findall(line))


class Configuration:
    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def load_config(maindir):
        """loads the configuration saved in the yaml file in order to use or update the content in a separate file"""
        from shutil import copyfile

        try:
            with open(os.path.join(maindir, 'config_imagingTB.yaml'), 'r') as yfile:
                cfg = yaml.safe_load(yfile)
        except FileNotFoundError:
            warnings.warn("No valid configuration file was found. Trying to creat a file named: "
                          "config_imagingTB.yaml in ROOTDIR from default values")
            filename_def = os.path.join(maindir, 'private') + 'config_imagingTBdef.yaml'
            if os.path.isfile(filename_def):
                copyfile(filename_def, os.path.join(ROOTDIR, 'config_imagingTBdef.yaml'))
            else:
                warnings.warn_explicit("No default configuration file found. Please make sure this is available in the "
                                       "./private folder or download from {}".format(GITHUB))
                return

        return cfg

    @staticmethod
    def save_config(rootdir, cfg):
        """saves the configuration in a yaml file in order to use or update the content in a separate file"""

        with open(os.path.join(rootdir, 'config_imagingTB.yaml'), 'wb') as settings_mod:
            yaml.safe_dump(cfg, settings_mod, default_flow_style=False,
                           explicit_start=True, allow_unicode=True, encoding='utf-8')


class Imaging:
    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def set_viewer(viewer_opt):
        """Sets the viewer_option in the configuration file"""
        from dependencies import ROOTDIR

        cfg = Configuration.load_config(ROOTDIR)
        if viewer_opt == 'itk-snap':  # to-date, only one viewer is available. May be changed in a future
            if not cfg["folders"]["path2itksnap"]:
                cfg["folders"]["path2itksnap"] = LocationCheck.FileLocation.itk_snap_check(ROOTDIR)
                Configuration.save_config(ROOTDIR, cfg)

        return cfg["folders"]["path2itksnap"]

    @staticmethod
    def load_imageviewer(path2viewer, file_names, suffix=''):
        """loads selected NIFTI-files in imageviewer"""

        if not file_names:
            Output.msg_box(txt="The provided list with NIFTI-files is empty, please double-check",
                           title="No NIFTI-files provided")
            return

        if sys.platform == ('linux' or 'linux2'):
            if len(file_names) == 1:
                cmd = ["itksnap", "-g", file_names[0]]
            else:
                cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]
        elif sys.platform == 'macos' or sys.platform == 'darwin':
            if len(file_names) == 1:
                cmd = ["itksnap", "-g", file_names[0]]
            else:
                cmd = ["itksnap", "-g", file_names[0], "-o", *file_names[1:]]

            # TODO: add sudo /Applications/ITK-SNAP.app/Contents/bin/install_cmdl.sh to the setup
            # TODO: change ITK-SNAP so that it does not freeze the entire script
            # TODO: remove other viewers apart from itksnap and options for win systems

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
            Output.msg_box(text="Viewers other than ITK-SNAP are not implemented!",
                           title="Unknown viewer")

    @staticmethod
    def resampleANTs(mm_spacing, ANTsImageObject, file_id, method):
        """Function aiming at rescaling imaging to a resolution specified somewhere, e.g. in the cfg-file"""
        import math, ants

        resolution = [mm_spacing] * 3
        if not all([math.isclose(float(mm_spacing), x, abs_tol=10 ** -2) for x in ANTsImageObject.spacing]):
            print('\tImage spacing {:.4f}x{:.4f}x{:.4f} unequal to specified value ({}mm). '
                  '\n\t\tRescaling {}'.format(ANTsImageObject.spacing[0], ANTsImageObject.spacing[1],
                                              ANTsImageObject.spacing[2], mm_spacing, file_id))

            if len(ANTsImageObject.spacing) == 4:
                resolution.append(ANTsImageObject.spacing[-1])
            elif len(ANTsImageObject.spacing) > 4:
                msg_box(text='\tSequences of >4 dimensions are not possible.', title='Too many dimensions')

            resampled_image = ants.resample_image(ANTsImageObject, resolution, use_voxels=False, interp_type=method)
            ants.image_write(resampled_image, filename=file_id)
        else:
            print('\tImage spacing for sequence: {} is {:.4f}x{:.4f}x{:.4f} as specified in options, '
                  '\t -> proceeding!'.format(file_id, ANTsImageObject.spacing[0], ANTsImageObject.spacing[1],
                                             ANTsImageObject.spacing[2]))
            resampled_image = ANTsImageObject

        return resampled_image

    def display_files_in_viewer(inputdir, regex2include, regex2exclude, selected_subjects='', viewer='itk-snap'):
        """Routine intended to provide generic way of displaying a batch of data using the viewer selected"""
        import glob

        cfg = Configuration.load_config(ROOTDIR)

        if not selected_subjects:
            Output.msg_box(text="No folder selected. To proceed, please indicate what folder to process.",
                           title="No subject selected")
            return
        elif len(selected_subjects) > 1:
            Output.msg_box(text="Please select only one folder to avoid loading too many images",
                           title="Too many subjects selected")
            return
        else:
            image_folder = os.path.join(cfg["folders"]["nifti"], selected_subjects[0])

        viewer_path = Imaging.set_viewer(viewer)  # to-date, only one viewer is available. May be changed in a future
        regex_complete = regex2include
        regex2exclude = [''.join('~{}'.format(y)) for y in regex2exclude]
        regex_complete += regex2exclude

        r = re.compile(r"^~.*")
        excluded_sequences = [x[1:] for x in list(filter(r.match, regex_complete))]
        included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

        all_files = glob.glob(image_folder + '/**/*.nii',
                              recursive=True)  # returns all available NIFTI-files in the folder including subfolders
        all_files2 = FileOperations.list_files_in_folder(inputdir=image_folder, contains='',
                                                         suffix='nii')  # TODO does it compare with old function?

        file_IDs = []
        [file_IDs.append(x) for x in all_files for y in included_sequences
         if re.search(r'\w+{}.'.format(y), x, re.IGNORECASE)]

        Imaging.load_imageviewer(viewer_path, sorted(file_IDs))

    @staticmethod
    def create_brainmask(input_folder, registered_images):
        """this function import antspynet in order to obtain a probabilistic brain mask for the T1 imaging"""
        import ants, antspynet
        filename_brainmask = os.path.join(input_folder, 'brainmask_T1.nii')

        brainmask = antspynet.brain_extraction(image=registered_images, verbose=False)
        ants.image_write(image=brainmask, filename=filename_brainmask)

    @staticmethod
    def sphere(diameter):
        """function defining binary matrix which represents a 3D sphere which may be used as structuring element"""

        struct = np.zeros((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
        x, y, z = np.indices((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
        mask = (x - diameter) ** 2 + (y - diameter) ** 2 + (z - diameter) ** 2 <= diameter ** 2
        struct[mask] = 1

        return struct.astype(np.bool)


class FileOperations:
    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def create_folder(foldername):
        """creates folder if not existent"""

        if not os.path.isdir(foldername):
            os.mkdir(foldername)

    @staticmethod
    def list_folders(inputdir, prefix='subj', files2lookfor='NIFTI'):
        """takes folder and lists all available subjects in this folder according to some filter given as [prefix]"""

        list_all = [name for name in os.listdir(inputdir)
                    if (os.path.isdir(os.path.join(inputdir, name)) and prefix in name)]

        if list_all == '':
            list_subj = 'No available subjects, please make sure {}-files are present and correct ' \
                        '"prefix" is given'.format(files2lookfor)
        else:
            list_subj = set(list_all)

        return list_subj

    @staticmethod
    def list_files_in_folder(inputdir, contains='', suffix='nii', entire_path=False, subfolders=True):
        """returns a list of files within a folder (including subfolders"""

        if subfolders:
            allfiles_in_folder = glob.glob(os.path.join(inputdir + '/**/*.' + suffix), recursive=True)
        else:
            allfiles_in_folder = glob.glob(inputdir + '/*.' + suffix)

        if not contains:
            filelist = [file_id for file_id in allfiles_in_folder]
        else:
            filelist = [file_id for file_id in allfiles_in_folder if any(y in file_id for y in contains)]

        if not entire_path:
            filelist = [os.path.split(x)[1] for x in filelist]

        return filelist

    @staticmethod
    def get_filelist_as_tuple(inputdir, subjects):
        """create a list of all available files in a folder and returns tuple together with the name of subject"""

        allfiles = []
        [allfiles.extend(
            zip(glob.glob(os.path.join(inputdir, x + "/*")), [x] * len(glob.glob(os.path.join(inputdir, x + "/*")))))
            for x in subjects]

        return allfiles

    @staticmethod
    def inner_join(a, b):
        """from: https://stackoverflow.com/questions/31887447/how-do-i-merge-two-lists-of-tuples-based-on-a-key"""

        L = a + b
        L.sort(key=itemgetter(1))  # sort by the first column
        for _, group in groupby(L, itemgetter(1)):
            row_a, row_b = next(group), next(group, None)
            if row_b is not None:  # join
                yield row_b[0:1] + row_a  # cut 1st column from 2nd row

    @staticmethod
    def set_wdir_in_config(cfg, foldername, init=False):
        """Generic function setting the working directory (e.g. DICOM, nifti, etc."""
        from PyQt5.QtWidgets import QFileDialog
        text2display = 'directory of nii-files' if foldername=='nifti' else 'dicom-folder'
        cfg['folders'][foldername] = ''
        while cfg['folders'][foldername] == '':
            if init:
                Output.msg_box(text="Directory not found, please select different one.", title="Directory not found")
            selected_directory = QFileDialog.getExistingDirectory(caption="Please select the {}".format(text2display))
            cfg['folders'][foldername] = selected_directory
            init = True
            Configuration.save_config(ROOTDIR, cfg)

        return selected_directory


class MatlabEquivalent:
    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def accumarray(a, accmap):
        """ from https://stackoverflow.com/questions/16856470/is-there-a-matlab-accumarray-equivalent-in-numpy"""

        ordered_indices = np.argsort(accmap)
        ordered_accmap = accmap[ordered_indices]
        _, sum_indices = np.unique(ordered_accmap, return_index=True)
        cumulative_sum = np.cumsum(a[ordered_indices])[sum_indices - 1]

        result = np.empty(len(sum_indices), dtype=a.dtype)
        result[:-1] = cumulative_sum[1:]
        result[-1] = cumulative_sum[0]
        result[1:] = result[1:] - cumulative_sum[1:]

        return result
