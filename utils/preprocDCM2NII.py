#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import sys
import time
import utils.HelperFunctions as HF
import pandas as pds


class PreprocessDCM:
    """in this class a functions are defined which aim at extracting data from DICOM files to nifti images and for basic
    work on these"""

    def __init__(self, _folderlist):
        self.logfile = True
        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

        self.cfg = HF.LittleHelpers.load_config(rootdir)
        self.DCM2NIIX_ROOT = os.path.join(self.cfg["folders"]["rootdir"], 'ext', 'dcm2niix')

        if not os.path.isdir(self.cfg["folders"]["dicom"]):
            text = "Please indicate a correct folder in the main GUI"
            title = "Wrong folder!"
            HF.LittleHelpers.msg_box(text=text, title=title)
        else:
            self.inputdir = self.cfg["folders"]["dicom"]

        subjlist = self.create_subjlist(_folderlist)

        if not os.path.isdir(self.DCM2NIIX_ROOT):
            text = "Extracting imaging data from DICOM files unsuccessful because of wrong folder for dcm2niix."
            title = "Wrong folder!"
            HF.LittleHelpers.msg_box(text=text, title=title)
            return

        if not os.path.isdir(self.cfg["folders"]["nifti"]):
            print('\nDirectory for output is invalid; assuming same base and creating folder named "NIFTI" therein!')
            self.outdir = os.path.join(os.path.split(self.inputdir)[0], "NIFTI")
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        else:
            self.outdir = self.cfg["folders"]["nifti"]

        lastsubj = self.get_index_nifti_folders(self.outdir, prefix=self.cfg["folders"]["prefix"])
        self.convert_dcm2nii(subjlist, last_folder=int(lastsubj))

    def convert_dcm2nii(self, dicomfolders, last_folder):
        """extracts DICOM-data and saves them to NIFTI-files to be further processed"""

        mods = ['CT', 'MRT']
        if self.logfile:
            log_filename = os.path.join(self.cfg["folders"]["rootdir"], 'logs',
                                        "log_DCM2NII_" + time.strftime("%Y%m%d-%H%M%S"))
        else:
            log_filename = os.devnull

        subj_list_save = os.path.join(self.cfg["folders"]["nifti"], 'subjdetails.csv')
        if not os.path.isfile(subj_list_save):
            df_save = pds.DataFrame(columns=['name', 'folder'])
        else:
            df_save = pds.read_csv(subj_list_save, sep='\t')

        print('\nExtracting DICOM files for {} subject(s)'.format(len(dicomfolders)))
        HF.LittleHelpers.printProgressBar(0, len(dicomfolders) * len(mods), prefix='Progress:', suffix='Complete',
                                          length=50, decimals=1)
        idx = 0
        start_time_complete = time.time()
        for k, name in enumerate(dicomfolders, start=last_folder + 1):
            start_time_subject = time.time()
            data_temp = {'name' : name, 'folder': self.cfg["folders"]["prefix"] + str(k)}
            df_save = df_save.append(data_temp, ignore_index=True)

            dcm2niix_bin = self.get_dcm2niix(self.DCM2NIIX_ROOT)
            subj_outdir = os.path.join(self.outdir, self.cfg["folders"]["prefix"] + str(k))
            if not os.path.isdir(subj_outdir):
                os.mkdir(subj_outdir)

            keptfiles, deletedfiles = ([] for i in range(2))
            for m in mods:
                idx += 1
                input_folder_name = os.path.join(self.inputdir, name + m)
                input_folder_files = [f.path for f in os.scandir(input_folder_name) if (f.is_dir() and "100" in f.path)]
                if type(input_folder_files) == list:
                    input_folder_files = ''.join(input_folder_files)

                orig_stdout = sys.stdout
                sys.stdout = open(log_filename, 'w')
                subprocess.call([dcm2niix_bin,
                                 '-b', self.cfg["preprocess"]["dcm2nii"]["BIDSsidecar"][0],
                                 '-z', self.cfg["preprocess"]["dcm2nii"]["OutputCompression"][0],
                                 '-f', self.cfg["preprocess"]["dcm2nii"]["OutputFileStruct"],
                                 '-o', subj_outdir,
                                 '-w', str(self.cfg["preprocess"]["dcm2nii"]["NameConflicts"]),
                                 '-v', str(self.cfg["preprocess"]["dcm2nii"]["Verbosity"]),
                                 '-x', str(self.cfg["preprocess"]["dcm2nii"]["ReorientCrop"]),
                                 input_folder_files],
                                stdout=sys.stdout, stderr=subprocess.STDOUT)
                sys.stdout.close()
                sys.stdout = orig_stdout
                HF.LittleHelpers.printProgressBar(idx, len(dicomfolders) * len(mods), prefix='Progress:',
                                                  suffix='Complete', length=50, decimals=1)
                files_kept, files_deleted = self.select_sequences(subj_outdir)
                keptfiles.extend(files_kept)
                deletedfiles.extend(files_deleted)

            # Functions creating/updating pipeline log, which document individually all steps along with settings
            log_text = "{} files successfully converted: {}, \n\nand {} deleted: {}.\nDuration: {:.2f} secs" \
                .format(len(set(keptfiles)),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(keptfiles)))),
                        len(set(deletedfiles)),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(deletedfiles)))),
                        time.time() - start_time_subject)
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=data_temp["folder"], module='dcm2nii',
                                             opt=self.cfg["preprocess"]["dcm2nii"], project="")

        df_save.to_csv(subj_list_save, index=False, header=True, sep='\t')
        print('\nIn total, a list of {} subjects was created'.format(len(dicomfolders)))
        print('\nData extraction took {:.2f}secs.'.format(time.time() - start_time_complete), end='', flush=True)
        print()

    def select_sequences(self, subj_outdir):
        """Function enabling user to select only some sequences; this is particularly helpful when a stack of files
        is extracted from DICOM-data"""
        import glob
        import itertools

        list_sequences = self.cfg["preprocess"]["dcm2nii"]["IncludeFiles"].split(",")
        if not list_sequences:
            return
        else:
            keeplist_temp = []
            keeplist_temp.extend([glob.glob(os.path.join(subj_outdir, '*' + x.lower() + '*'))
                                  for x in list_sequences]) # for generalisability, upper-/lowercase tested sequentially
            keeplist_temp.extend([glob.glob(os.path.join(subj_outdir, '*' + x.upper() + '*'))
                                  for x in list_sequences])
            keeplist = list(itertools.chain.from_iterable(keeplist_temp))
            keeplist = [array for array in keeplist if ("SCOUT" not in array and "REFORMATION" not in array)]
            allfiles = glob.glob(os.path.join(subj_outdir, '*'))
            files_to_delete = list(set(allfiles) - set(keeplist))
            [os.remove(x) for x in files_to_delete]

        return keeplist, files_to_delete

    @staticmethod
    def get_dcm2niix(DCM2NIIX_ROOT):
        """ returns the location of the dcm2niix files for different operating systems; this is identical with:
        https://github.com/devhliu/intelligentLiver/blob/master/dcmconv/dcm2niix.py"""

        if sys.platform == 'win32':
            dcm2niix_bin = os.path.join(DCM2NIIX_ROOT, 'dcm2niix.exe')
        elif sys.platform == 'linux':
            dcm2niix_bin = "dcm2niix"
        elif sys.platform == 'macos':
            dcm2niix_bin = os.path.join(DCM2NIIX_ROOT, 'dcm2niix_mac')
        else:
            print('Chris Rordens dcm2niix routine not found, please make sure it is available.', end='', flush=True)
            dcm2niix_bin = False

        return dcm2niix_bin

    @staticmethod
    def create_subjlist(folderlist):
        """creates a set of subjects; Especially, different modalities are summarised as one but only if filenames are
        identical"""
        allnames = [re.findall("[a-z][^A-Z]*", x) for x in folderlist]
        available_subj = set(x for letters in allnames for x in letters)

        return available_subj

    @staticmethod
    def get_index_nifti_folders(niftidir, prefix='subj'):
        """this function provides the index of the last processed subject. That is, in a list of consecutive recordings,
         nifti-folder is opened and already available folders are counted"""

        list_dirs = [subdir for subdir in os.listdir(niftidir) if (prefix in subdir and
                                                                   os.path.isdir(os.path.join(niftidir, subdir)) and
                                                                   len(os.listdir(os.path.join(niftidir, subdir))) > 0)]
        if not list_dirs:
            idx = 0
        else:
            idx = list_dirs[-1].split(prefix)[-1]

        return idx
