#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

import pandas as pds

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration, FileOperations


class PreprocessDCM:
    """in this class a functions are defined which aim at extracting data from DICOM files to nifti images and for basic
    work on these"""

    # TODO the following problem need to be addressed: 1) the way the DICOM folders are detected is somewhat
    #  arbitrary (=hard-coded, cf. line 99f.) this needs a fix

    def __init__(self, _folderlist):
        self.logfile = True
        self.cfg = Configuration.load_config(ROOTDIR)
        self.DCM2NIIX_ROOT = os.path.join(ROOTDIR, 'ext', 'dcm2niix')

        if not os.path.isdir(self.cfg['folders']['dicom']):
            Output.msg_box(text="Please indicate a correct folder in the main GUI", title="Wrong folder")
        else:
            self.inputdir = self.cfg['folders']['dicom']

        subjlist = self.create_subjlist(_folderlist)

        if not os.path.isdir(self.DCM2NIIX_ROOT):
            Output.msg_box(text="Extracting imaging data from DICOM-files not successful because of wrong "
                                "folder for 'dcm2niix'.", title="Wrong folder!")
            return

        if not os.path.isdir(self.cfg['folders']['nifti']):
            print("\nDirectory for output is invalid; assuming same base and creating folder named 'nifti' therein!")
            self.outdir = os.path.join(os.path.split(self.inputdir)[0], 'nifti')
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)
        else:
            self.outdir = self.cfg['folders']['nifti']

        lastsubj = self.get_index_nifti_folders(self.outdir, prefix=self.cfg['folders']['prefix'])

        self.create_csv_subjlist(subjlist, int(lastsubj))
        self.convert_dcm2nii(subjlist, last_idx=int(lastsubj))

    def convert_dcm2nii(self, dicomfolders, last_idx):
        """extracts DICOM-data and saves them to NIFTI-files to be further processed; for this, the multiprocessing
        toolbox is used for a use of all available cores"""

        print('\nExtracting DICOM files for {} subject(s)'.format(len(dicomfolders)))
        start_multi = time.time()
        status = mp.Queue()
        processes = [mp.Process(target=self.dcm2niix_multiprocessing,
                                args=(name, ind, self.get_dcm2niix(self.DCM2NIIX_ROOT), last_idx,
                                      len(dicomfolders), status)) for ind, name in enumerate(dicomfolders, start=1)]
        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                subj_name, modality, no_subj, total_subj = status.get()
                print("Extracting DICOM-folders ({}) for subj{}: {} (subject {}/{})"
                      .format(modality, no_subj, subj_name, no_subj, total_subj))
            time.sleep(0.1)

        for p in processes:
            p.join()

        print("\nIn total, a list of {} subjects was created".format(len(dicomfolders)))
        print("\nData extraction took {:.2f}secs.".format(time.time() - start_multi), end='', flush=True)
        print()

    def dcm2niix_multiprocessing(self, name_subj, no_subj, dcm2niix_bin, last_idx, total_subj, status):
        """function intended to provide multiprocessing approach to speed up extraction of DICOM data to nifti files"""

        modalities = ['CT', 'MRI']
        if self.logfile:
            log_filename = os.path.join(ROOTDIR, 'logs', 'log_DCM2NII_' + str(no_subj + last_idx) +
                                        time.strftime("%Y%m%d-%H%M%S"))
        else:
            log_filename = os.devnull

        subj_outdir = os.path.join(self.outdir, self.cfg['folders']['prefix'] + str(no_subj + last_idx))
        FileOperations.create_folder(subj_outdir)
        start_time_subject = time.time()
        keptfiles, deletedfiles = ([] for _ in range(2))

        for mod in modalities:
            status.put((name_subj, mod, no_subj, total_subj))
            input_folder_name = os.path.join(self.inputdir, name_subj + mod)
            input_folder_files = [f.path for f in os.scandir(input_folder_name)
                                  if (f.is_dir() and ('100' in f.path or '001' in f.path or 'DICOM' in f.path))]

            orig_stdout = sys.stdout
            sys.stdout = open(log_filename, 'w')
            for folder in input_folder_files:
                subprocess.call([dcm2niix_bin,
                                 '-a', 'y',  # anonimisation of DICOM data
                                 '-b', self.cfg['preprocess']['dcm2nii']['BIDSsidecar'][0],
                                 '-z', self.cfg['preprocess']['dcm2nii']['OutputCompression'][0],
                                 '-f', self.cfg['preprocess']['dcm2nii']['OutputFileStruct'],
                                 '-o', subj_outdir,
                                 '-w', str(self.cfg['preprocess']['dcm2nii']['NameConflicts']),
                                 '-v', str(self.cfg['preprocess']['dcm2nii']['Verbosity']),
                                 '-x', str(self.cfg['preprocess']['dcm2nii']['ReorientCrop']),
                                 folder],
                                stdout=sys.stdout, stderr=subprocess.STDOUT)

            sys.stdout.close()
            sys.stdout = orig_stdout

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
        Output.logging_routine(text=Output.split_lines(log_text), cfg=self.cfg,
                               subject=self.cfg['folders']['prefix'] + str(no_subj), module='dcm2nii',
                               opt=self.cfg['preprocess']['dcm2nii'], project="")

    def create_csv_subjlist(self, subjlist, first_index):
        """ this function creates a csv-files which aims at providing information about which name/pseudnonym, etc.
         corresponds to the respective subject in the NIFTI-folders"""

        subjlist_filename = os.path.join(self.cfg['folders']['nifti'], 'subjdetails.csv')
        if not os.path.isfile(subjlist_filename):
            df_save = pds.DataFrame(columns=['name', 'folder'])
        else:
            df_save = pds.read_csv(subjlist_filename, index_col=0, sep='\t')

        dtemp = {'name': [], 'folder': []}
        for idx, name in enumerate(subjlist, start=first_index + 1):
            dtemp['name'].append(name)
            dtemp['folder'].append(self.cfg['folders']['prefix'] + str(idx))

        df_new = pds.DataFrame(dtemp, columns=['name', 'folder'])
        df_save = df_save.append(df_new)
        df_save.reset_index(drop=True)
        df_save.to_csv(subjlist_filename, index=True, header=True, sep='\t')

    def select_sequences(self, subj_outdir):
        """Function enabling user to select only some sequences; this is particularly helpful when a stack of files
        is extracted from DICOM-folder"""
        import glob

        allsequences = glob.glob(os.path.join(subj_outdir, '*'))
        regex_complete = self.cfg['preprocess']['dcm2nii']['IncludeFiles'].split(",")
        regex_complete += ['~SCOUT', '~AAH', '~REFORMATION']

        r = re.compile(r"^~.*")
        excluded_sequences = [x[1:] for x in list(filter(r.match, regex_complete))]
        included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

        if not included_sequences:
            included_sequences = ['.']

        keeplist, black_list = [], []
        [keeplist.append(x) for x in glob.glob(os.path.join(subj_outdir, '*')) for y in included_sequences
         if re.search(r'\w+{}.'.format(y), x, re.IGNORECASE)]

        [black_list.append(x) for x in keeplist for y in excluded_sequences
         if re.search(r'\w+{}.'.format(y), x, re.IGNORECASE)]

        keeplist = list(set(keeplist) - set(black_list))
        files_to_delete = list(set(allsequences) - set(keeplist))

        [os.remove(x) for x in files_to_delete]
        return keeplist, files_to_delete

    @staticmethod
    def get_dcm2niix(DCM2NIIX_ROOT):
        """ returns the location of the dcm2niix files for different operating systems; this is identical with:
        https://github.com/devhliu/intelligentLiver/blob/master/dcmconv/dcm2niix.py"""

        if sys.platform == 'linux':
            dcm2niix_bin = os.path.join(DCM2NIIX_ROOT, 'dcm2niix')
        elif sys.platform == 'macos' or sys.platform == 'darwin':
            dcm2niix_bin = os.path.join(DCM2NIIX_ROOT, 'macos', 'dcm2niix')
        else:
            print("Chris Rordens dcm2niix routines not found, please make sure they are available.", end='', flush=True)
            dcm2niix_bin = False

        return dcm2niix_bin

    @staticmethod
    def create_subjlist(folderlist):
        """creates set of subjects; Especially, different modalities are summarised as one iff filenames identical;
        extraction of DICOM is somehow different to rest of lists because of possibly >1 folders per subj., so that
        separate function is necessary instead of the one in [HelperFunctions.py]"""

        allnames = [re.findall("[a-z][^A-Z]*", x) for x in folderlist]
        available_subj = set(x for letters in allnames for x in letters)

        return available_subj

    @staticmethod
    def get_index_nifti_folders(niftidir, prefix='subj'):
        """function providing indices of the last processed subject. That is, in a list of consecutive recordings,
         nifti-folder is opened and already available folders are counted"""

        list_dirs = [subdir for subdir in os.listdir(niftidir) if (prefix in subdir and
                                                                   os.path.isdir(os.path.join(niftidir, subdir)) and
                                                                   len(os.listdir(os.path.join(niftidir, subdir))) > 0)]
        try:
            all_endings = [int(re.search(r'({})(\w+)'.format(prefix), x).group(2)) for x in list_dirs]
            idx = 0 if not all_endings else sorted(all_endings)[-1]
        except ValueError:
            idx = 0

        return idx
