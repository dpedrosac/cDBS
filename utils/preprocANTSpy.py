#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import ants
import time
import utils.HelperFunctions as HF
import multiprocessing as mp
import glob
import math
import sys
import traceback
import shutil
from itertools import groupby
from operator import itemgetter
from dependencies import ROOTDIR


class ProcessANTSpy:
    """this class contains all functions used by the ANTsPy Toolbox; in general the multiprocessing routines are
    implemented aiming at making the code as efficient and quick as possible."""

    def __init__(self):
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)

    def N4BiasCorrection(self, subjects):
        """function performing N4BiasCorrection according to N.J. Tustison, ..., and J.C. Gee.
        "N4ITK: Improved N3 Bias Correction" IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010."""

        print('\nDebiasing imaging of {} subject(s)'.format(len(subjects)))
        # allfiles = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=subjects)

        # TODO: replace this part with the one in HelperFunction (see above) and make it as generic as possible as
        #  it may be needed for other parts of the toolbox
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
                results.append([pool.apply_async(func=self.N4BiasCorrection_multiprocessing,  #self.func_wrapper_debug
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

    def N4BiasCorrection_multiprocessing(self, filename, subj, input_folder):
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
        if 0 < float(self.cfg["preprocess"]["Registration"]["resample_spacing"]) != any(bcorr_image.spacing[0:2]):
            bcorr_image = self.ANTrescale_image(bcorr_image,
                                                float(self.cfg["preprocess"]["Registration"]["resample_spacing"]),
                                                int(self.cfg["preprocess"]["Registration"]["resample_method"]))

            # TODO: if this works, delete the next few lines
            #resolution = [float(self.cfg["preprocess"]["Registration"]["resample_spacing"])] * 3
            #if len(bcorr_image.spacing) > 3:
            #    resolution.append(bcorr_image.spacing[-1])

            #bcorr_image = ants.resample_image(bcorr_image, (resolution),
            #                                  use_voxels=False,
            #                                  interp_type=int(self.cfg["preprocess"]["Registration"]["resample_method"]))


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
    def ANTrescale_image(image_sequence, mm_spacing, method):
        """recales the imaging to a resolution specified in the options"""

        resolution = [mm_spacing] * 3
        if len(image_sequence.spacing) > 3:
            resolution.append(image_sequence.spacing[-1])
        resampled_image = ants.resample_image(image_sequence, (resolution), use_voxels=False, interp_type=method)

        return resampled_image

    @staticmethod
    def check_bias_correction(allfiles):
        """function allowing to check for the difference due to correction. Specifically, """

    def ANTsCoregisterCT2MRI(self, subjects, fixed_image='bc_t1'):
        """function performing Coregistration of postoperative CT to preopertive MRI for further steps in the analysis
         pipeline"""

        print('\nStarting Coregistration for {} subject(s)'.format(len(subjects)))
        input_folder = self.cfg["folders"]["nifti"]
        allfiles = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=subjects)

        regex_complete = ['CT_', '{}_'.format(fixed_image.upper())]
        included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

        file_ID_CT, file_ID_MRI = [], []
        [file_ID_CT.append(x) for x in allfiles
         if re.search(r'\w+{}.'.format(included_sequences[0]), x[0], re.IGNORECASE) and x[0].endswith('.nii')]
        # TODO There may be a problem at this point, especially when > 1 run is performed!

        [file_ID_MRI.append(x) for x in allfiles
         if re.search(r'\w+(?!_).({}).'.format(included_sequences[1]), x[0], re.IGNORECASE) and x[0].endswith('.nii')]
        if not file_ID_MRI:
            HF.msg_box(text="No MRI sequence with boas correction found. Please double-check",
                       title="Preprocessed MRI-sequence missing")
            return
        fileIDs = list(self.inner_join(file_ID_CT, file_ID_MRI))

        start_multi = time.time()
        with mp.Pool(int(math.floor(mp.cpu_count() / 2))) as pool:
            try:
                [pool.apply_async(func=self.func_wrapper_debug,  # self.N4Bias_correction_multiprocessing
                                  args=(filename_mri, filename_ct, no_subj,
                                        os.path.join(self.cfg["folders"]["nifti"], no_subj)))
                 for filename_mri, filename_ct, no_subj in fileIDs]
            finally:
                pool.close()
                pool.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for no_subj in subjects:
            allfiles_subj = [os.path.split(files_subj)[1] for files_subj, subj_no in fileIDs if subj_no == no_subj]
            log_text = "{} files successfully processed: {}, \n\n Mean Duration per subject: {:.2f} secs" \
                .format(len(set(allfiles_subj)),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(allfiles_subj)))),
                        (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(no_subj), module='N4BiasCorrection',
                                             opt=self.cfg["preprocess"]["ANTsN4"], project="")

        print('\nIn total, a list of {} subjects was processed'.format(len(subjects)))
        print('\nCT registration took {:.2f}secs. overall'.format(time.time() - start_multi), end='', flush=True)
        print()

    def ANTsCoregisterCT2T1_multiprocessing(self, fixed_sequence, moving_sequence, subj, input_folder):
        """Does the Coregistration between twi images taking advantage of multicores, so that multiple subjects
        can be processed in parallel; For that a list of tuples including the entire filename and the subject
        to be processed are entered"""

        print("Start registering {} (moving file) to {} (fixed file), using the ANTs routines".format(fixed_sequence,
                                                                                                      moving_sequence))
        previous_registrations = glob.glob(os.path.join(input_folder, subj + "/" +
                                                        self.cfg["preprocess"]["registration"]["prefix"] +
                                                        os.path.split(moving_sequence)[1] + 'run*.nii'))
        if not previous_registrations:
            print('No previous registration found, starting with first run')
            run = 1
        else:
            allruns = [re.search(r'\w+(run)([\d.]+)', x[0]).group(1) for x in previous_registrations]
            lastrun = int(sorted(allruns)[-1])
            print('Found {} registrations before. Do you want to affine registration or begin register all over '
                  'again?'.format(str(lastrun))) # TODO a message box is needed and all registrations mus be deleted in case start from scratch is desired
            # filename_registered =
            run = lastrun + 1

        filename_save = os.path.join(input_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                     str(run) + '_' + os.path.split(moving_sequence)[1])
        print("Process: {}; Co-Registration {}; filenames: {}(f) and {}(m)".format(mp.current_process().name, subj,
                                                                                   os.path.split(fixed_sequence)[1],
                                                                                   os.path.split(moving_sequence)[1]))
        image_to_process = dict()
        for idx, file_id in enumerate([fixed_sequence, moving_sequence]):
            sequence = ants.image_read(file_id)
            spacing = float(self.cfg["preprocess"]["registration"]["resample_spacing"])
            if 0 < float(spacing) != any(sequence.spacing[0:2]):
                print('Image spacing {:.4f}x{:.4f}x{:.4f} unequal to specified value ({}mm). '
                      'Rescaling {}'.format(sequence.spacing[0], sequence.spacing[1], sequence.spacing[2],
                                            spacing, file_id),
                      end='', flush=True)
                if idx == 0:
                    image_to_process[idx] = self.ANTrescale_image(sequence, spacing,
                                                       int(self.cfg["preprocess"]["registration"]["resample_method"]))
                else:
                    image_to_process[idx] = self.ANTrescale_image(sequence, spacing,
                                                       int(self.cfg["preprocess"]["registration"]["resample_method"]))
                print('Done!')
            else:
                print('Image spacing for sequence: {} is {:.4f}x{:.4f}x{:.4f} as specified in options. '
                      'Continuing!'.format(file_id, sequence.spacing[0],sequence.spacing[1],sequence.spacing[2]))

        skull_strip = 0
        if skull_strip:
            print('Implementation stil required!') # TODO: implement the brain extraction routine from ANT

        # TODO: Two-step approach seems most promising and it should depend on whether registration was already
        #  performed or not. Therefore filename recognition is necessary and a loop which runs Rigid and Affine
        #  registration sequentially and changes the metric (see lead-dbs for example)

        metric = [('mattes', 'mattes'), ('GC', 'mattes'), ('mattes', 'mattes'), ('mattes', 'GC'), ('mattes', 'GC')]
        # Start with Co-Registration of CT and T1 imaging
        if run == 1:
            # Rigid Registration (1st step)
            # TODO winsorize not available and interpolation (Linear) not possible; besides, combination of both steps unclear
            reg_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                          type_of_transform='Rigid', grad_step=.1,
                                          aff_metric=metric[run][0], aff_sampling=32, aff_random_sampling_rate=.25,
                                          aff_iterations = [1000,500,250,100,0], # self.cfg["preprocess"]["registration"]["max_iterations"],
                                          aff_shrink_factors=(12, 8, 4, 2, 1),
                                          aff_smoothing_sigmas=(4, 3, 2, 1, 1),
                                          initial_transform="[%s,%s,1]" % (fixed_sequence, moving_sequence),
                                          verbose=True) # TODO: Last row sort of redundant. It shoould be checked if input is identical in default version vs. this way (ants.registration.interface.py, lines 320, 331, 332, 340)

            # Affine registration (2nd step))
            reg_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                          type_of_transform='Affine', grad_step=.1,
                                          aff_metric=metric[run][0], aff_sampling=32, aff_random_sampling_rate=.25,
                                          aff_iterations = [1000,500,250,100,0],#self.cfg["preprocess"]["registration"]["max_iterations"],
                                          aff_shrink_factors=(12, 8, 4, 2, 1),
                                          aff_smoothing_sigmas=(4, 3, 2, 1, 1),
                                          initial_transform="[%s,%s,1]" % (fixed_sequence, moving_sequence),
                                          verbose=False)
        elif run == 6:
            print("If registration fails after five times, a different approach should be pursued!")
            return
        else:
            # TODO incliude initial transform!
            # Rigid Registration (1st step)
            reg_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                          type_of_transform='Rigid', grad_step=.1,
                                          aff_metric=metric[run][0], aff_sampling=32, aff_random_sampling_rate=.25,
                                          aff_iterations=self.cfg["preprocess"]["registration"]["max_iterations"],
                                          aff_shrink_factors=(12, 8, 4, 2, 1),
                                          aff_smoothing_sigmas=(4, 3, 2, 1, 1),
                                          verbose=False)

            # Affine registration (2nd step)
            reg_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                          type_of_transform='Rigid', grad_step=.1,
                                          aff_metric=metric[run][0], aff_sampling=32, aff_random_sampling_rate=.25,
                                          aff_iterations=self.cfg["preprocess"]["registration"]["max_iterations"],
                                          aff_shrink_factors=(12, 8, 4, 2, 1),
                                          aff_smoothing_sigmas=(4, 3, 2, 1, 1),
                                          verbose=False)

        ants.image_write(reg_image, filename=filename_save)

        # Old versions of the file are moved to the debug-folder
        debug_folder = os.path.join(input_folder, "debug")
        if run > 1:
            if not os.path.isdir(debug_folder):
                os.mkdir(debug_folder)
            filename_previous = os.path.join(input_folder, self.cfg["preprocess"]["Registration"]["prefix"] + 'run_' +
                                             str(lastrun) + os.path.split(moving_sequence)[1])
            shutil.move(filename_previous, debug_folder)


    def func_wrapper_debug(self, filename_mri, filename_ct, subj, input_folder):
        """this is just a wrapper intended to debug the code, that is to callback possible errors"""
        try:
            self.ANTsCoregisterCT2T1_multiprocessing(filename_mri, filename_ct, subj, input_folder)
            #self.N4Bias_correction_multiprocessing(filename, subj, input_folder)
        except Exception as e:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    @staticmethod
    def inner_join(a, b):
        """adapted from: https://stackoverflow.com/questions/31887447/how-do-i-merge-two-lists-of-tuples-based-on-a-key"""
        L = a + b
        L.sort(key=itemgetter(1)) # sort by the first column
        for _, group in groupby(L, itemgetter(1)):
            row_a, row_b = next(group), next(group, None)
            if row_b is not None: # join
                yield row_b[0:1] + row_a  # cut 1st column from 2nd row
