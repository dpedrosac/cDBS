#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ants
import time
import utils.HelperFunctions as HF
import multiprocessing as mp
import glob
import math
import sys
import traceback


class ProcessANTSpy:
    """this class contains all functions used by the ANTsPy Toolbox; in general the multiprocessing routines are
    implemented aiming at making the code as efficient and quick as possible."""

    def __init__(self):
        rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        self.cfg = HF.LittleHelpers.load_config(rootdir)

    def N4BiasCorrection(self, subjects):
        """function performing N4BiasCorrection according to N.J. Tustison, ..., and J.C. Gee.
        "N4ITK: Improved N3 Bias Correction" IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010."""

        print('\nDebiasing imaging of {} subject(s)'.format(len(subjects)))

        allfiles = []
        [allfiles.extend(list(zip(glob.glob(os.path.join(self.cfg["folders"]["nifti"], x + "/*")),
                                  [x] * len(glob.glob(os.path.join(self.cfg["folders"]["nifti"], x + "/*"))))))
         for x in subjects]

        allfiles = [file_tuple for file_tuple in allfiles if ('CT' not in file_tuple[0]
                                                              and file_tuple[0].endswith(".nii")
                                                              and self.cfg["preprocess"]["ANTsN4"]["prefix"]
                                                              not in file_tuple[0])]

        file_id_DTI = [file_tuple for file_tuple in allfiles
                       if ('CT' not in file_tuple[0]
                           and self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in file_tuple[0]
                           and file_tuple[0].endswith('.nii')
                           and self.cfg["preprocess"]["ANTsN4"]["prefix"] not in file_tuple[0])]

        file_id_noDTI = [file_tuple for file_tuple in allfiles
                         if ('CT' not in file_tuple[0]
                             and not self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]
                                     in file_tuple[0]
                             and file_tuple[0].endswith('.nii') and self.cfg["preprocess"]["ANTsN4"]["prefix"]
                             not in file_tuple[0])]

        seq = ''  # Only for debugging purposes
        if not seq:
            file_IDs = allfiles
        elif seq == 'struct':
            file_IDs = list(set(allfiles) - set(file_id_DTI))
        elif seq == 'dwi':
            file_IDs = list(set(allfiles) - set(file_id_noDTI))

        start_multi = time.time()
        with mp.Pool(int(math.floor(mp.cpu_count() / 2))) as pool:
            results = []
            try:
                results.append([pool.apply_async(func=self.func_wrapper_debug,  # self.N4Bias_correction_multiprocessing
                                                 args=(name_file, no_subj, os.path.join(self.cfg["folders"]["nifti"],
                                                                                        no_subj)))
                                for name_file, no_subj in file_IDs])
            finally:
                pool.close()
                pool.join()

                # Functions creating/updating pipeline log, which document individually all steps along with settings
                for no_subj in subjects:
                    allfiles_subj = [os.path.split(files_subj)[1] for files_subj, subj_no in file_IDs if
                                     subj_no == no_subj]
                    log_text = "{} files successfully processed: {}, \n\n Mean Duration per subject: {:.2f} secs" \
                        .format(len(set(allfiles_subj)),
                                '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(allfiles_subj)))),
                                (time.time() - start_multi) / len(subjects))
                    HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                                     subject=str(no_subj), module='N4BiasCorrection',
                                                     opt=self.cfg["preprocess"]["ANTsN4"], project="")

                print('\nIn total, a list of {} subjects was processed'.format(len(subjects)))
                print('\nOverall, bias correction took {:.2f}secs.'.format(time.time() - start_multi))
                print('Done!')

        debug = False
        if debug:
            for r in results:
                r.get()

    def func_wrapper_debug(self, filename, subj, input_folder):
        """this is just a wrapper intended to debug the code, that is to callback possible errors"""
        try:
            self.N4Bias_correction_multiprocessing(filename, subj, input_folder)
        except Exception as e:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))
            #return e

    def N4Bias_correction_multiprocessing(self, filename, subj, input_folder):
        """Does the Bias correction taken advantage of the multicores, so that multiple subjects can be processed in
        parallel; For that a list of tuples including the entire filename and the subject to be processed are entered"""

        filename_save = os.path.join(input_folder, self.cfg["preprocess"]["ANTsN4"]["prefix"] +
                                     os.path.split(filename)[1])
        print("Process: {}; Running debiasing for {}; filename: {}".format(mp.current_process().name, subj,
                                                                           os.path.split(filename)[1]))
        # Start with N4 Bias correction for sequences specified before
        original_image = ants.image_read(os.path.join(input_folder, filename))
        rescaler_nonneg = ants.contrib.RescaleIntensity(10, 100)
        if self.cfg["preprocess"]["ANTsN4"]["denoise"] == 'yes':
            original_image = ants.denoise_image(image=original_image, noise_model='Rician')

        min_orig, max_orig = original_image.min(), original_image.max()
        if not os.path.split(filename)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]):
            original_image_nonneg = rescaler_nonneg.transform(original_image)
        else:
            original_image_nonneg = original_image

        bcorr_image = ants.utils.n4_bias_field_correction(original_image_nonneg,
                                                          mask=None,
                                                          shrink_factor=
                                                          self.cfg["preprocess"]["ANTsN4"]["shrink-factor"],
                                                          convergence=
                                                          {'iters': self.cfg["preprocess"]["ANTsN4"]["convergence"], # it may be necessary to adapt this according to the number of shrinkage factors
                                                           'tol': self.cfg["preprocess"]["ANTsN4"]["threshold"]},
                                                          spline_param=
                                                          self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"],
                                                          verbose=False,
                                                          weight_mask=None)

        if not os.path.split(filename)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]):
            rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
            bcorr_image = rescaler.transform(bcorr_image)

        # Resample image
        if float(self.cfg["preprocess"]["Registration"]["resample_spacing"]) > 0:
            resolution = [float(self.cfg["preprocess"]["Registration"]["resample_spacing"])] * 3
            if len(bcorr_image.spacing) > 3:
                resolution.append(bcorr_image.spacing[-1])

            bcorr_image = ants.resample_image(bcorr_image, (resolution),
                                              use_voxels=False,
                                              interp_type=int(self.cfg["preprocess"]["Registration"]["resample_method"]))

        ants.image_write(bcorr_image, filename=filename_save)

        # A difference between both images is saved for debugging purposes
        debug_folder = os.path.join(input_folder, "debug")
        if not os.path.isdir(debug_folder):
            os.mkdir(debug_folder)

        diff_image_save = os.path.join(debug_folder, "diff_biasCorr_" +
                                       os.path.split(filename)[1])
        diff_image = original_image - bcorr_image
        ants.image_write(diff_image, filename=diff_image_save)

    @staticmethod
    def check_bias_correction(allfiles):
        """function allowing to check for the difference due to correction. Specifically, """

    def ANTsCoregisterWrapper(self, subjects, template='', fixed_image='t1'):
        """function performing Coregistration of MRI imaging to template in order to create standardises images which
        may be compared among each other"""

        print('\nStarting Coregistration for {} subject(s)'.format(len(subjects)))
        input_folder = self.cfg["folders"]["nifti"]

        all_files = []
        [all_files.extend(glob.glob(os.path.join(input_folder, x + "/" + self.cfg["preprocess"]["ANTsN4"]["prefix"]
                                                 + "*"))) for x in subjects]

        file_id_CT = [sequence_name for sequence_name in all_files if 'CT' in sequence_name]
        file_id_DTI = [sequence_name for sequence_name in all_files
                       if ('CT' not in sequence_name
                           and self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in sequence_name
                           and sequence_name.endswith('.nii')
                           and not os.split(sequence_name)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["prefix"]))]

        file_id_noDTI = [sequence_name for sequence_name in all_files
                         if ('CT' not in sequence_name
                             and not self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in sequence_name
                             and sequence_name.endswith('.nii')
                             and not os.split(sequence_name)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["prefix"]))]

        seq = 'struct'  # Only for debugging purposes
        if not seq:
            file_IDs = all_files
        elif seq == 'struct':
            file_IDs = list(set(all_files) - set(file_id_DTI))
        elif seq == 'dwi':
            file_IDs = list(set(all_files) - set(file_id_noDTI))

        # TODO find the t1 and define it as fixed image!

        start_multi = time.time()
        with mp.Pool(int(math.floor(mp.cpu_count() / 2))) as pool:
            try:
                [pool.apply_async(func=self.func_wrapper_debug,  # self.N4Bias_correction_multiprocessing
                                  args=(name_file, no_subj, os.path.join(self.cfg["folders"]["nifti"], no_subj)))
                 for name_file, no_subj in file_IDs]
            finally:
                pool.close()
                pool.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for no_subj in subjects:
            allfiles_subj = [os.path.split(files_subj)[1] for files_subj, subj_no in file_IDs if subj_no == no_subj]
            log_text = "{} files successfully processed: {}, \n\n Mean Duration per subject: {:.2f} secs" \
                .format(len(set(allfiles_subj)),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(allfiles_subj)))),
                        (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(no_subj), module='N4BiasCorrection',
                                             opt=self.cfg["preprocess"]["ANTsN4"], project="")

        print('\nIn total, a list of {} subjects was processed'.format(len(subjects)))
        print('\nBias correction took {:.2f}secs. overall'.format(time.time() - start_multi), end='', flush=True)
        print()
