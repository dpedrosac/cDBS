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
import numpy as np
import shutil
from itertools import groupby
from operator import itemgetter
from dependencies import ROOTDIR
from PyQt5.QtWidgets import QMessageBox


class ProcessANTSpy:
    """this class contains all functions used by the ANTsPy Toolbox; in general the multiprocessing routines are
    implemented aiming at making the code as efficient and quick as possible."""

    def __init__(self):
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)

    def N4BiasCorrection(self, subjects):
        """N4BiasCorrection according to N.J. Tustison, ..., and J.C. Gee.
        "N4ITK: Improved N3 Bias Correction" IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010."""

        print('\nDebiasing imaging of {} subject(s)'.format(len(subjects)))
        allfiles = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=subjects)
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

        # debug possibility, uncomment if needed
        #   for r in results:
        #       r.get()

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
         if re.search(r'\w+{}.'.format(included_sequences[0]), x[0], re.IGNORECASE) and x[0].endswith('.nii')
         and 'run' not in x[0]]

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
            files_processed = [(os.path.split(file_moving)[1], os.path.split(file_fixed)[1])
                               for file_fixed, file_moving, subj_no in fileIDs if subj_no == no_subj]
            log_text = "{} successfully registered to \n{}, \n\n Mean Duration per subject: {:.2f} " \
                       "secs".format(files_processed[0][0], files_processed[0][1], (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(no_subj), module='CT2MRIRegistration',
                                             opt=self.cfg["preprocess"]["registration"], project="")

        print('\nIn total, a list of {} subjects was processed'.format(len(subjects)))
        print('\nCT registration took {:.2f}secs. overall'.format(time.time() - start_multi), end='', flush=True)
        print()

    # ==============================    Multiprocessing functions   ==============================

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

        bcorr_image = ants.utils.n4_bias_field_correction(original_image_nonneg, mask=None,
                                                          shrink_factor=
                                                          self.cfg["preprocess"]["ANTsN4"]["shrink-factor"],
                                                          convergence=
                                                          {'iters': self.cfg["preprocess"]["ANTsN4"]["convergence"],
                                                           # it may be necessary to adapt this according to the number of shrinkage factors
                                                           'tol': self.cfg["preprocess"]["ANTsN4"]["threshold"]},
                                                          spline_param=
                                                          self.cfg["preprocess"]["ANTsN4"]["bspline-fitting"],
                                                          verbose=False, weight_mask=None)

        if not os.path.split(filename)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]):
            rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
            bcorr_image = rescaler.transform(bcorr_image)

        # Resample image
        #        if 0 < float(self.cfg["preprocess"]["Registration"]["resample_spacing"]) != any(bcorr_image.spacing[0:2]):
        if not all([math.isclose(float(self.cfg["preprocess"]["Registration"]["resample_spacing"]),
                                 x, abs_tol=10 ** -2) for x in bcorr_image.spacing]):
            bcorr_image = self.ANTrescale_image(bcorr_image,
                                                float(self.cfg["preprocess"]["Registration"]["resample_spacing"]),
                                                int(self.cfg["preprocess"]["Registralen(image_sequence.spacing) == 4tion"]["resample_method"]))

        ants.image_write(bcorr_image, filename=filename_save)

        # Difference between both images is saved for debugging purposes
        debug_folder = os.path.join(input_folder, "debug")
        if not os.path.isdir(debug_folder):
            os.mkdir(debug_folder)

        diff_image = original_image - bcorr_image
        ants.image_write(diff_image,
                         filename=os.path.join(debug_folder, "diff_biasCorr_" + os.path.split(filename)[1]))

    def ANTsCoregisterCT2T1_multiprocessing(self, fixed_sequence, moving_sequence, subj, input_folder):
        """Does the Co-Registration between twi images taking advantage of multicores, so that multiple subjects
        can be processed in parallel; For that a list of tuples including the entire filename and the subject
        to be processed are entered"""

        print("Registering {} (moving file) to {} (fixed file), using ANTsPy".format(fixed_sequence, moving_sequence))
        previous_registrations = glob.glob(os.path.join(input_folder + "/" +
                                                        self.cfg["preprocess"]["registration"]["prefix"] + 'run*' +
                                                        os.path.split(moving_sequence)[1]))
        if not previous_registrations:
            print('No previous registration found, starting with first run')
            run = 1
        else:
            allruns = [re.search(r'\w+(run)([\d.]+)', x).group(2) for x in previous_registrations]
            lastrun = int(sorted(allruns)[-1])
            filename_lastrun = os.path.join(input_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                            str(lastrun) + '_' + os.path.split(moving_sequence)[1])
            run = lastrun + 1

        filename_save = os.path.join(input_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                     str(run) + '_' + os.path.split(moving_sequence)[1])
        log_filename = os.path.join(ROOTDIR, 'logs', "log_CT2MRI_RegisterANTs_{}_run_{}_".format(subj, str(run)) +
                                    time.strftime("%Y%m%d-%H%M%S") + '.txt')

        print("Process: {}; Co-Registration {}; filenames: {}(f) and {}(m)".format(mp.current_process().name, subj,
                                                                                   os.path.split(fixed_sequence)[1],
                                                                                   os.path.split(moving_sequence)[1]))
        image_to_process = dict()
        for idx, file_id in enumerate([fixed_sequence, moving_sequence]):
            sequence = ants.image_read(file_id)
            spacing = float(self.cfg["preprocess"]["registration"]["resample_spacing"])

            if not all ([math.isclose(float(spacing), x, abs_tol=10**-2) for x in sequence.spacing]):
                print('Image spacing {:.4f}x{:.4f}x{:.4f} unequal to specified value ({}mm). '
                      'Rescaling {}'.format(sequence.spacing[0], sequence.spacing[1], sequence.spacing[2],
                                            spacing, file_id),
                      end='', flush=True)
                image_to_process[idx] = self.ANTrescale_image(sequence, spacing,
                                                              int(self.cfg["preprocess"]["registration"]
                                                                  ["resample_method"]))
                ants.image_write(image_to_process[idx], filename=file_id)
                print(' ... Done!')
            else:
                image_to_process[idx] = sequence
                print('Image spacing for sequence: {} is {:.4f}x{:.4f}x{:.4f} as specified in options. '
                      'Continuing!'.format(file_id, sequence.spacing[0],sequence.spacing[1],sequence.spacing[2]))

        skull_strip = 0
        if skull_strip:
            print('Implementation still required!') # TODO: implement the brain extraction routine from ANTsPyNET

        # Start with Co-Registration of CT and T1 imaging
        fixed_sequence_filename = fixed_sequence
        if run == 1:
            moving_sequence_filename = moving_sequence
            metric = 'mattes'

        elif run == 6:
            HF.msg_box(text="If registration fails after five times, different approach may be necessary!",
                       title="Registration failed too many times")
            return

        else:
            moving_sequence_filename = filename_lastrun
            metric = 'GC'

        if self.cfg["preprocess"]["registration"]["default_registration"] == 'yes':
            registered_images = self.default_registration(image_to_process, fixed_sequence_filename,
                                                          moving_sequence_filename, input_folder + '/', log_filename,
                                                          metric=metric)
        else:
            registered_images = self.custom_registration(image_to_process, fixed_sequence_filename,
                                                         moving_sequence_filename, input_folder + '/', log_filename,
                                                         run)

        ants.image_write(registered_images['warpedmovout'], filename=filename_save)

        debug_folder = os.path.join(input_folder, "debug")
        if not os.path.isdir(debug_folder):
            os.mkdir(debug_folder)

        filename_save = os.path.join(debug_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                     str(run) + '_' + os.path.split(fixed_sequence)[1])
        ants.image_write(registered_images['warpedfixout'], filename=filename_save)

        # Old versions of the file are moved to the debug-folder
        if run > 1:
            filename_previous = os.path.join(input_folder, self.cfg["preprocess"]["Registration"]["prefix"] + 'run_' +
                                             str(lastrun) + os.path.split(moving_sequence)[1])
            filename_dest = os.path.join(debug_folder, self.cfg["preprocess"]["Registration"]["prefix"] + 'run_' +
                                             str(lastrun) + os.path.split(moving_sequence)[1])
            shutil.move(filename_previous, filename_dest)

    def default_registration(self, image_to_process, sequencename1, sequencename2, inputfolder, log_filename, metric='mattes'):
        """runs the default version of the ANTs registration routine, that is a Rigid transformation, an affine trans-
        formation and a symmetric normalisation (SyN). Further options available using the cmdline """

        #sys.stdout = open(log_filename, 'w+') # TODO: logging not working; Must be rewritten possibly using logging module?!?
        transform = self.cfg["preprocess"]["registration"]["default_registration_metric"]
        registered_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                             type_of_transform=transform, grad_step=.1,
                                             aff_metric=metric,
                                             outprefix=inputfolder,
                                             initial_transform="[%s,%s,1]" % (sequencename1, sequencename2),
                                             verbose=True)

        return registered_image

    def custom_registration(self, image_to_process, sequencename1, sequencename2, inputfolder, log_filename, runs):
        """runs a custom registration using the routines provided in ANTsPy. This entire function results from the code
         in interface.py from (https://github.com/ANTsX/ANTsPy/blob/master/ants/registration/interface.py) and was
         extended according to the documentation for the original ANTsRegistration.cxx script
         https://github.com/ANTsX/ANTs/blob/7ed2b4b264885f1056d21b225760af1463450510/Examples/antsRegistration.cxx
         Careful use is advised as not all sanity checks are included and not all options are available automatically"""

        name = ['fixed', 'moving']
        for idx in image_to_process:
            if np.sum(np.isnan(image_to_process[idx].numpy())) > 0:
                raise ValueError("{} image has NaNs - replace these".format(name[idx]))
            image_to_process[idx] = image_to_process[idx].clone("float")
            if idx == 0:
                inpixeltype = image_to_process[0].pixeltype

        warpedmovout = image_to_process[0].clone()
        warpedfixout = image_to_process[1].clone()

        fixed_pointer = ants.utils.get_pointer_string(image_to_process[0])
        moving_pointer = ants.utils.get_pointer_string(image_to_process[1])

        wfo_pointer = ants.utils.get_pointer_string(warpedfixout)
        wmo_pointer = ants.utils.get_pointer_string(warpedmovout)

        processed_args = []
        # TODO: add option for winsorize and for use_histogram or similar
        with open(os.path.join(ROOTDIR, "utils",
                               self.cfg["preprocess"]["registration"]["custom_registration_file"]), "r") as f:
            for line in f:
                processed_args.append(line.strip())

        replace_items = [('filename_fixed', sequencename1), ('filename_moving', sequencename2),
                         ('fixed_ANTsimage_process', fixed_pointer), ('moving_ANTsimage_process', moving_pointer),
                         ('input_folder', inputfolder),
                         ('warpedmovout_process', wmo_pointer), ('warpedfixout_process', wfo_pointer)]

        for item, replacement in replace_items:
            processed_args=[re.sub(r'[*]' + item + r'[*]', replacement, x) for x in processed_args]

        orig_stdout = sys.stdout
        sys.stdout = open(log_filename, 'w')
        libfn = ants.utils.get_lib_fn("antsRegistration")
        libfn(processed_args)
        sys.stdout = orig_stdout
        sys.stdout.close()

        afffns = glob.glob(inputfolder + "*" + "[0-9]GenericAffine.mat")
        fwarpfns = glob.glob(inputfolder + "*" + "[0-9]Warp.nii.gz")
        iwarpfns = glob.glob(inputfolder + "*" + "[0-9]InverseWarp.nii.gz")

        # print(afffns, fwarpfns, iwarpfns)
        if len(afffns) == 0:
            afffns = ""
        if len(fwarpfns) == 0:
            fwarpfns = ""
        if len(iwarpfns) == 0:
            iwarpfns = ""

        alltx = sorted(glob.glob(inputfolder + "*" + "[0-9]*"))
        findinv = np.where([re.search("[0-9]InverseWarp.nii.gz", ff) for ff in alltx])[0]
        findfwd = np.where([re.search("[0-9]Warp.nii.gz", ff) for ff in alltx])[0]

        if len(findinv) > 0:
            fwdtransforms = list(reversed([ff for idx, ff in enumerate(alltx) if idx != findinv[0]]))
            invtransforms = [ff for idx, ff in enumerate(alltx) if idx != findfwd[0]]
        else:
            fwdtransforms = list(reversed(alltx))
            invtransforms = alltx

        registered_image = {"warpedmovout": warpedmovout.clone(inpixeltype),
                            "warpedfixout": warpedfixout.clone(inpixeltype),
                            "fwdtransforms": fwdtransforms,
                            "invtransforms": invtransforms}

        return registered_image

    def func_wrapper_debug(self, filename_mri, filename_ct, subj, input_folder):
        """this is just a wrapper intended to debug the code, that is to callback possible errors"""
        try:
            self.ANTsCoregisterCT2T1_multiprocessing(filename_mri, filename_ct, subj, input_folder)
            #self.N4Bias_correction_multiprocessing(filename, subj, input_folder)
        except Exception as e:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    @staticmethod
    def ANTrescale_image(image_sequence, mm_spacing, method):
        """Function aiming at rescaling imaging to a resolution specified somewhere, e.g. in the cfg-file"""

        resolution = [mm_spacing] * 3
        if len(image_sequence.spacing) == 4:
            resolution.append(image_sequence.spacing[-1])
        elif len(image_sequence.spacing) > 4:
            HF.msg_box(text='Sequences of >4 dimensions are not possible.', title='Too many dimensions')
        resampled_image = ants.resample_image(image_sequence, (resolution), use_voxels=False, interp_type=method)

        return resampled_image

    @staticmethod
    def inner_join(a, b):
        """adapted from: https://stackoverflow.com/questions/31887447/how-do-i-merge-two-lists-of-tuples-based-on-a-key"""
        L = a + b
        L.sort(key=itemgetter(1)) # sort by the first column
        for _, group in groupby(L, itemgetter(1)):
            row_a, row_b = next(group), next(group, None)
            if row_b is not None: # join
                yield row_b[0:1] + row_a  # cut 1st column from 2nd row
