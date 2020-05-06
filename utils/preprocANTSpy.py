#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import ants
import sys
import re
import time
import utils.HelperFunctions as HF
import glob
import subprocess
from PyQt5.QtWidgets import QMessageBox


class ProcessANTSpy:
    def __init__(self):
        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.cfg = HF.LittleHelpers.load_config(rootdir)

    def N4BiasCorrection(self, subjects, output_folder_prefix=''):
        """function performing N4BiasCorrection according to N.J. Tustison, ..., and J.C. Gee.
        "N4ITK: Improved N3 Bias Correction" IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010."""

        if not output_folder_prefix:
            output_folder = os.path.join(self.cfg["folders"]["nifti"])
        else:
            output_folder = os.path.join(self.cfg["folders"]["nifti"], output_folder_prefix)

        for idx, subj_name in enumerate(subjects):
            subj_folder = os.path.join(self.cfg["folders"]["nifti"], subj_name)
            all_files = [filename for filename in os.listdir(subj_folder)
                         if ('CT' not in filename and filename.endswith(".nii"))]

            file_id_DTI = [filename for filename in os.listdir(subj_folder) if
                           ('CT' not in filename and self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in filename)
                           and filename.endswith('.nii')]

            file_id_noDTI = [filename for filename in os.listdir(subj_folder) if
                             ('CT' not in filename and not self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in filename)
                             and filename.endswith('.nii')]

            # ===================================   Structural imaging  ===================================

            print("Starting with the preprocessing of structural data, i.e. no DTI imaging is included")
            HF.LittleHelpers.printProgressBar(0, len(file_id_noDTI), prefix='Progress:', suffix='Complete',
                                              length=50, decimals=1)

            startTime = time.time()
            for idx, seq in enumerate(file_id_noDTI):
                HF.LittleHelpers.printProgressBar(idx, len(file_id_noDTI), prefix='Progress:', suffix='Complete',
                                                  length=50, decimals=1)
                rescaler = ants.contrib.RescaleIntensity(10, 100)
                denoised_image = ants.image_read(os.path.join(subj_folder, seq))
                #denoised_image = ants.denoise_image(image=original_image, noise_model='Rician')

                min_orig, max_orig = denoised_image.min(), denoised_image.max()
                denoised_image_nonneg = rescaler.transform(denoised_image)

                bc_image = ants.utils.n4_bias_field_correction(denoised_image_nonneg,
                                                               mask=None,
                                                               shrink_factor=
                                                               self.cfg["preprocess"]["ANTsN4"]["shrink-factor"],
                                                               convergence=
                                                               {'iters': self.cfg["preprocess"]["ANTsN4"]["convergence"],
                                                                'tol': self.cfg["preprocess"]["ANTsN4"]["threshold"]},
                                                               spline_param=
                                                               self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"],
                                                               verbose=True,
                                                               weight_mask=None)

                rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
                bc_image = rescaler.transform(bc_image)
                save_filename = os.path.join(output_folder, subj_folder,
                                             self.cfg["preprocess"]["ANTsN4"]["prefix"] + seq)
                ants.image_write(bc_image, filename=save_filename)
                ants.image_write()

            opt_arg = '\nRunning Bias Field Correction using the N4 B-Spline algorithm from the ANTsPy repository for {} ' \
                      'files(s): \n{}'.format(len(all_files), ''.join("\t{}\n".format(f) for f in all_files))
            # TODO: Add this to the log-file according to the HelperFunctions
            print(opt_arg)

            HF.LittleHelpers.printProgressBar(0, len(all_files), prefix='Progress:', suffix='Complete',
                                              length=50, decimals=1)

            if len(all_files) > 1:
                print('\nIn total, 1 file was processed')
            else:
                print('\nIn total, {} files were processed'.format(len(all_files)))

            # Functions creating/updating pipeline log, which document individually all steps along with settings
            log_text = "{} files successfully processed; {}; \n\nand {} deleted: {}.\nDuration: {:.2f} secs" \
                .format(len(all_files), '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(all_files))),
                        time.time() - startTime)


            time_processing = "\nBias correction took {:.2f}secs.".format(time.time() - startTime)
            opt_arg = opt_arg.join(time_processing)
            # TODO here data should be passed to the logging file in helper functions. Besides a new function is needed to
            #  make sure data qiuality can be checked in ITK SNAP and a button is required in the second tab of GUI.
            print(time_processing, end='', flush=True)
            print()

        @staticmethod
        def check_bias_correction(original, corrected):
            """function allowing to check for the difference due to correction. Specifically, """

        # scales = get the original scale


