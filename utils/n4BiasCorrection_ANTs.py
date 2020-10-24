#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import os
import re
import time

import ants

from utils.HelperFunctions import Output, Configuration, Imaging, FileOperations, MatlabEquivalent
from dependencies import ROOTDIR
from ants.utils import n4_bias_field_correction as n4biascorr

# TODO: 1. Bias correction for DTI results in enormous images
class BiasCorrection:
    """this class contains all functions used by the ANTsPy Toolbox; in general the multiprocessing routines are
    implemented aiming at making the code as efficient and quick as possible."""

    def __init__(self):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.verbose = 0

    def N4BiasCorrection(self, subjects):
        """N4BiasCorrection according to N.J. Tustison, ..., and J.C. Gee.
        "N4ITK: Improved N3 Bias Correction" IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010."""

        print('\nDebiasing imaging of {} subject(s)'.format(len(subjects)))
        allfiles = FileOperations.get_filelist_as_tuple(inputdir=self.cfg['folders']['nifti'], subjects=subjects)
        strings2exclude = ['CT', self.cfg['preprocess']['ANTsN4']['prefix'], 'reg_run', 'Mask']

        allfiles = [x for x in allfiles if x[0].endswith('.nii') and not
        any(re.search(r'\w+(?!_).({})|^({})\w+.'.format(z, z), os.path.basename(x[0]), re.IGNORECASE)
            for z in strings2exclude)]

        file_id_DTI = [x for x in allfiles if (x[0].endswith('.nii') and self.cfg['preprocess']['ANTsN4']['dti_prefix']
                                               in x[0]) and not
                       any(re.search(r'\w+(?!_).({})|^({})\w+.'.format(z, z), os.path.basename(x[0]), re.IGNORECASE)
                           for z in strings2exclude)]

        file_id_noDTI = [x for x in allfiles if (x[0].endswith('.nii') and not
        self.cfg['preprocess']['ANTsN4']['dti_prefix'] in x[0]) and not
                         any(re.search(r'\w+(?!_).({})|^({})\w+.'.format(z, z), os.path.basename(x[0]), re.IGNORECASE)
                             for z in strings2exclude)]

        seq = 'struct'  # For debugging purposes
        if not seq:
            fileIDs = allfiles
        elif seq == 'struct':
            fileIDs = list(set(allfiles) - set(file_id_DTI))
        elif seq == 'dwi':
            fileIDs = list(set(allfiles) - set(file_id_noDTI))

        start_multi = time.time()
        status = mp.Queue()
        processes = [mp.Process(target=self.N4BiasCorrection_multiprocessing,
                                args=(name_file, no_subj, os.path.join(self.cfg['folders']['nifti'], no_subj), status))
                     for name_file, no_subj in fileIDs]

        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                process, no_subj, filename = status.get()
                print("Process: {}; Debiasing {}, filename: {}".format(process, no_subj, filename))
            time.sleep(0.1)

        for p in processes:
            p.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for subjID in subjects:
            allfiles_subj = [os.path.split(files_subj)[1] for files_subj, subj_no in fileIDs if subj_no == subjID]
            log_text = "{} files successfully processed (@{}): {}, \n\n Mean Duration per subject: {:.2f} secs" \
                .format(len(set(allfiles_subj)), time.strftime("%Y%m%d-%H%M%S"),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(allfiles_subj)))),
                        (time.time() - start_multi) / len(subjects))
            Output.logging_routine(text=Output.split_lines(log_text), cfg=self.cfg,
                                   subject=str(subjID), module='N4BiasCorrection',
                                   opt=self.cfg['preprocess']['ANTsN4'], project="")

        print('\nIn total, a list of {} subject(s) was processed \nOverall, bias correction took '
              '{:.1f} secs.'.format(len(subjects), time.time() - start_multi))

    def N4BiasCorrection_multiprocessing(self, file2rename, subj, input_folder, status):
        """Does the Bias correction taking advantage of the multicores, so that multiple subjects can be processed in
        parallel; For that a list of tuples including the entire filename and the subject to be processed are entered"""

        status.put(tuple([mp.current_process().name, subj, os.path.split(file2rename)[1]]))
        filename_save = os.path.join(input_folder, self.cfg['preprocess']['ANTsN4']['prefix'] +
                                     os.path.split(file2rename)[1])

        # Start with N4 Bias correction for sequences specified before
        original_image = ants.image_read(os.path.join(input_folder, file2rename))
        rescaler_nonneg = ants.contrib.RescaleIntensity(10, 100)  # to avoid values <0 causing problems w/ log data
        if self.cfg['preprocess']['ANTsN4']['denoise'] == 'yes':  # takes forever and therefore not used by default
            original_image = ants.denoise_image(image=original_image, noise_model='Rician')

        min_orig, max_orig = original_image.min(), original_image.max()
        if not os.path.split(file2rename)[1].startswith(self.cfg['preprocess']['ANTsN4']['dti_prefix']):
            original_image_nonneg = rescaler_nonneg.transform(original_image)
        else:
            original_image_nonneg = original_image

        bcorr_image = n4biascorr(original_image_nonneg, mask=None,
                                 shrink_factor=self.cfg['preprocess']['ANTsN4']['shrink-factor'],
                                 convergence={'iters': self.cfg['preprocess']['ANTsN4']['convergence'],
                                              'tol': self.cfg['preprocess']['ANTsN4']['threshold']},
                                 spline_param=self.cfg['preprocess']['ANTsN4']['bspline-fitting'],
                                 verbose=bool(self.verbose), weight_mask=None)

        if not os.path.split(file2rename)[1].startswith(self.cfg['preprocess']['ANTsN4']['dti_prefix']):
            rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
            bcorr_image = rescaler.transform(bcorr_image)

        # difference between both images is saved for debugging purposes
        diff_image = original_image - bcorr_image
        FileOperations.create_folder(os.path.join(input_folder, "debug"))  # only creates folder if not present
        ants.image_write(diff_image, filename=os.path.join(input_folder, "debug", "diff_biasCorr_" +
                                                           os.path.split(file2rename)[1]))

        spacing = self.cfg['preprocess']['registration']['resample_spacing']
        bcorr_image = Imaging.resampleANTs(mm_spacing=spacing, ANTsImageObject=bcorr_image,
                                           file_id=filename_save, method=int(self.cfg['preprocess']
                                                                             ['registration']
                                                                             ['resample_method']))
        ants.image_write(bcorr_image, filename=filename_save)
