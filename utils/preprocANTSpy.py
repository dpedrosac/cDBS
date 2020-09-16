#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import ants
import time
import utils.HelperFunctions as HF
import multiprocessing as mp
import glob
import numpy as np
import shutil
from itertools import groupby
from operator import itemgetter
from dependencies import ROOTDIR


# TODO: 1. Bias correction for DTI results in enormous images, 2. second run of ANTsCoRegisterCT2MRI not working
class ProcessANTSpy:
    """this class contains all functions used by the ANTsPy Toolbox; in general the multiprocessing routines are
    implemented aiming at making the code as efficient and quick as possible."""

    def __init__(self):
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)
        self.verbose = 0

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
                             and not self.cfg["preprocess"]["ANTsN4"]["dti_prefix"] in file_tuple[0]
                             and file_tuple[0].endswith('.nii') and self.cfg["preprocess"]["ANTsN4"]["prefix"]
                             not in file_tuple[0])]

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
                                args=(name_file, no_subj, os.path.join(self.cfg["folders"]["nifti"], no_subj), status))
                     for name_file, no_subj in fileIDs]
        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                try:
                    process, no_subj, filename = status.get()
                    print("Process: {}; Debiasing {}, filename: {}".format(process, no_subj, filename))
                except mp.Queue.Empty:
                    break
            time.sleep(0.1)

        for p in processes:
            p.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for subjID in subjects:
            allfiles_subj = [os.path.split(files_subj)[1] for files_subj, subj_no in fileIDs if subj_no == subjID]
            log_text = "{} files successfully processed: {}, \n\n Mean Duration per subject: {:.2f} secs" \
                .format(len(set(allfiles_subj)),
                        '\n\t{}'.format('\n\t'.join(os.path.split(x)[1] for x in sorted(set(allfiles_subj)))),
                        (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(subjID), module='N4BiasCorrection',
                                             opt=self.cfg["preprocess"]["ANTsN4"], project="")

        print('\nIn total, a list of {} subject(s) was processed \nOverall, bias correction took '
              '{:.1f} secs.'.format(len(subjects), time.time() - start_multi))

    def N4BiasCorrection_multiprocessing(self, file2rename, subj, input_folder, status):
        """Does the Bias correction taking advantage of the multicores, so that multiple subjects can be processed in
        parallel; For that a list of tuples including the entire filename and the subject to be processed are entered"""

        status.put(tuple([mp.current_process().name, subj, os.path.split(file2rename)[1]]))
        filename_save = os.path.join(input_folder, self.cfg["preprocess"]["ANTsN4"]["prefix"] +
                                     os.path.split(file2rename)[1])

        # Start with N4 Bias correction for sequences specified before
        original_image = ants.image_read(os.path.join(input_folder, file2rename))
        rescaler_nonneg = ants.contrib.RescaleIntensity(10, 100)  # to avoid values <0 causing problems w/ log data
        if self.cfg["preprocess"]["ANTsN4"]["denoise"] == 'yes':  # takes forever and therefore not used by default
            original_image = ants.denoise_image(image=original_image, noise_model='Rician')

        min_orig, max_orig = original_image.min(), original_image.max()
        if not os.path.split(file2rename)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]):
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
                                                          verbose=self.verbose, weight_mask=None)

        if not os.path.split(file2rename)[1].startswith(self.cfg["preprocess"]["ANTsN4"]["dti_prefix"]):
            rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
            bcorr_image = rescaler.transform(bcorr_image)

        # difference between both images is saved for debugging purposes
        diff_image = original_image - bcorr_image
        HF.LittleHelpers.create_folder(os.path.join(input_folder, "debug"))  # only creates folder if not present
        ants.image_write(diff_image, filename=os.path.join(input_folder, "debug", "diff_biasCorr_" +
                                                           os.path.split(file2rename)[1]))

        spacing = self.cfg["preprocess"]["registration"]["resample_spacing"]
        bcorr_image = HF.resampleANTsImaging(mm_spacing=spacing, ANTsImageObject=bcorr_image, file_id=filename_save,
                                             method=int(self.cfg["preprocess"]["registration"]["resample_method"]))
        ants.image_write(bcorr_image, filename=filename_save)

    def ANTsCoregisterMRI2template(self, subjects):
        """function performing Co-Registration between MRI and specific template"""

        print('\nStarting Co-Registration for {} subject(s)'.format(len(subjects)))
        allfiles = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=subjects)
        templatefiles = glob.glob(os.path.join(ROOTDIR, 'ext', 'templates', self.cfg["preprocess"]["normalisation"]
        ["template_image"] + '/*'))

        sequences = self.cfg["preprocess"]["normalisation"]["sequences"].split(sep=',')

        fileIDs = []
        for idx, seqs in enumerate(sequences):
            file_template = [x for x in templatefiles if re.search(r'\w+.({}).'.format(seqs), x, re.IGNORECASE)] # template

            if not file_template:
                HF.msg_box(text='No teamplet could be found. Please make sure they are installled at ./ext/templates',
                            title='Template missing') # TODO install templates by default!
                return

            regex_complete = '{}{}'.format(self.cfg["preprocess"]["ANTsN4"]["prefix"], seqs)

            files_subj = [x for x in allfiles if
                          re.search(r'\w+(?!_).({}).'.format(regex_complete), x[0], re.IGNORECASE) and x[0].endswith('.nii')
                          and 'run' not in x[0]]
            file_list = [(file_template[0],file_id, subj) for file_id, subj in files_subj]

            fileIDs.extend(tuple(file_list))
        print(fileIDs)

        if not fileIDs:
            HF.msg_box(text="No MRI sequence with bias correction found. Please double-check",
                       title="Preprocessed MRI-sequence missing")
            return

        start_multi = time.time()
        status = mp.Queue()
        processes = [mp.Process(target=self.ANTsCoregisterMultiprocessing,
                                args=(filename_fixed, filename_moving, no_subj,
                                      os.path.join(self.cfg["folders"]["nifti"], no_subj), "registration", status))
                     for filename_fixed, filename_moving, no_subj in fileIDs]
        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                filename_fixed, filename_moving, no_subj = status.get()
                print("Registering {} (f) to {} (m), using ANTsPy".format(filename_fixed, filename_moving, no_subj))
            time.sleep(0.1)

        for p in processes:
            p.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for no_subj in subjects:
            files_processed = [(os.path.split(file_moving)[1], os.path.split(file_fixed)[1])
                               for file_fixed, file_moving, subj_no in fileIDs if subj_no == no_subj]
            log_text = "{} successfully normalised to \n{}, \n\n Mean Duration per subject: {:.2f} " \
                       "secs".format(files_processed[0][0], files_processed[0][1],
                                     (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(no_subj), module='MRI2templateRegistration',
                                             opt=self.cfg["preprocess"]["normalisation"], project="")

        print('\nIn total, a list of {} subjects was processed. \nCT registration took {:.2f}secs.'
              ' overall\n'.format(len(subjects), time.time() - start_multi))

    def ANTsCoregisterCT2MRI(self, subjects, input_folder, fixed_image='norm_run1_bc_t1'):
        """Coregistration of postoperative CT to preopertive MRI for further analyses in same space; before registration
        presence of normalised data is ensured to avoid redundancy"""

        print('\nStarting Coregistration for {} subject(s)'.format(len(subjects)))
        allfiles = HF.get_filelist_as_tuple(inputdir=input_folder, subjects=subjects)
        self.check_for_normalisation(subjects, fixed_image='')

        regex_complete = ['CT_', '{}_'.format(fixed_image.upper())]
        included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

        file_ID_CT, file_ID_MRI = ([] for i in range(2))
        [file_ID_CT.append(x) for x in allfiles
         if re.search(r'\w+{}.'.format(included_sequences[0]), x[0], re.IGNORECASE) and x[0].endswith('.nii')
         and 'run' not in x[0]]

        [file_ID_MRI.append(x) for x in allfiles # for simplicity written in a second line as regexp is slightly different
         if re.search(r'\w+(?!_).({}).'.format(included_sequences[1]), x[0], re.IGNORECASE) and x[0].endswith('.nii')]

        if not file_ID_MRI:
            HF.msg_box(text="Bias-corrected MRI not found. Please double-check", title="Preprocessed MRI unavailable")
            return
        fileIDs = list(self.inner_join(file_ID_CT, file_ID_MRI))

        # Start multiprocessing framework
        start_multi = time.time()
        status = mp.Queue()
        processes = [mp.Process(target=self.ANTsCoregisterMultiprocessing,
                                args=(filename_fixed, filename_moving, no_subj,
                                      os.path.join(self.cfg["folders"]["nifti"], no_subj), "registration", status))
                     for filename_fixed, filename_moving, no_subj in fileIDs]
        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                filename_fixed, filename_moving, no_subj = status.get()
                print("Registering {} (f) to {} (m), using ANTsPy".format(filename_fixed, filename_moving, no_subj))
            time.sleep(0.1)

        for p in processes:
            p.join()

        # Functions creating/updating pipeline log, which document individually all steps along with settings
        for no_subj in subjects:
            files_processed = [(os.path.split(file_moving)[1], os.path.split(file_fixed)[1])
                               for file_fixed, file_moving, subj_no in fileIDs if subj_no == no_subj]
            log_text = "{} successfully registered to \n{}, \n\n Mean Duration per subject: {:.2f} " \
                       "secs".format(files_processed[0][0], files_processed[0][1],
                                     (time.time() - start_multi) / len(subjects))
            HF.LittleHelpers.logging_routine(text=HF.LittleHelpers.split_lines(log_text), cfg=self.cfg,
                                             subject=str(no_subj), module='CT2MRIRegistration',
                                             opt=self.cfg["preprocess"]["registration"], project="")

        print('\nIn total, a list of {} subjects was processed CT registration took {:.2f}secs. '
              'overall'.format(len(subjects), time.time() - start_multi))

    # ==============================    Multiprocessing functions   ==============================
    def ANTsCoregisterMultiprocessing(self, fixed_sequence, moving_sequence, subj, input_folder, flag, status):
        """Does the Co-Registration between two images taking advantage of multicores, so that multiple subjects
        can be processed in parallel"""

        status.put(tuple([fixed_sequence, moving_sequence, subj]))
        if flag == "normalisation":
            prev_reg = ''
            files2rename = {'0GenericAffine.mat': '0GenericAffineMRI2template.mat',
                            '1InverseWarp.nii.gz': '1InverseWarpMRI2template.nii.gz',
                            '1Warp.nii.gz': '1WarpMRI2template.nii.gz'}
        else:
            prev_reg = glob.glob(os.path.join(input_folder + "/" + self.cfg["preprocess"][flag]["prefix"] + 'run*' +
                                                            os.path.split(moving_sequence)[1]))
            files2rename = {'0GenericAffine.mat': '0GenericAffineRegistration.mat',
                            '1InverseWarp.nii.gz': '1InverseWarpRegistration.nii.gz',
                            '1Warp.nii.gz': '1WarpRegistration.nii.gz'}

        if not prev_reg:
            print('No previous registration found, starting with first run')
            run = 1
        else:
            allruns = [re.search(r'\w+(run)([\d.]+)', x).group(2) for x in prev_reg]
            lastrun = int(sorted(allruns)[-1])
            filename_lastrun = os.path.join(input_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                            str(lastrun) + '_' + os.path.split(moving_sequence)[1])
            run = lastrun + 1

        filename_save = os.path.join(input_folder, self.cfg["preprocess"][flag]["prefix"] + 'run' +
                                     str(run) + '_' + os.path.split(moving_sequence)[1])

        log_filename = os.path.join(ROOTDIR, 'logs', "log_CT2MRI_RegisterANTs_{}_run_{}_".format(subj, str(run)) +
                                    time.strftime("%Y%m%d-%H%M%S") + '.txt')

        image_to_process = dict()
        for idx, file_id in enumerate([fixed_sequence, moving_sequence]):
            sequence = ants.image_read(file_id) # load data and resample images if necessary
            spacing = self.cfg["preprocess"]["registration"]["resample_spacing"]
            image_to_process[idx] = HF.resampleANTsImaging(mm_spacing=spacing,
                                                           ANTsImageObject=sequence, file_id=file_id,
                                                           method=int(self.cfg["preprocess"]["registration"]
                                                                      ["resample_method"]))
        fixed_sequence_filename = fixed_sequence
        if run == 1:
            moving_sequence_filename = moving_sequence
            metric = self.cfg["preprocess"]["registration"]["metric"][0]
        else: #TODO this part is not working whatsoever; requires some debugging. Returning to GuiMain",
            moving_sequence_filename = filename_lastrun
            self.cfg["preprocess"]["registration"]["default_registration"] = 'no'
            metric = self.cfg["preprocess"]["registration"]["metric"][0]

        if self.cfg["preprocess"]["registration"]["default_registration"] == 'yes':
            registered_images = self.default_registration(image_to_process, fixed_sequence_filename,
                                                          moving_sequence_filename, input_folder + '/', log_filename,
                                                          metric=metric)
        else:
            registered_images = self.custom_registration(image_to_process, fixed_sequence_filename,
                                                         moving_sequence_filename, input_folder + '/', log_filename,
                                                         run)
        for key in files2rename:
            name_of_file = glob.glob(os.path.join(input_folder, key))
            if name_of_file:
                os.rename(name_of_file[0],os.path.join(input_folder, files2rename[key]))

        ants.image_write(registered_images['warpedmovout'], filename=filename_save)
        HF.LittleHelpers.create_folder(os.path.join(input_folder, "debug")) #creates

        # 'Previous registrations' are moved to debug-folder
        if run > 1:
            filename_previous = os.path.join(input_folder, self.cfg["preprocess"][flag]["prefix"] + 'run_' +
                                             str(lastrun) + '_' + os.path.split(moving_sequence)[1])
            filename_dest = os.path.join(input_folder, "debug", self.cfg["preprocess"][flag]["prefix"] + 'RUNPREV_' +
                                         str(lastrun) + '_' + os.path.split(moving_sequence)[1])
            shutil.move(filename_previous, filename_dest)

        skull_strip = 1
        if skull_strip and 't1' in moving_sequence:
            import antspynet
            filename_brainmask = os.path.join(input_folder, 'brainmask_T1.nii')

            brainmask = antspynet.brain_extraction(image=registered_images['warpedmovout'], verbose=False)
            ants.image_write(image=brainmask, filename=filename_brainmask)


    def default_registration(self, image_to_process, sequence1, sequence2, inputfolder, log_filename, metric='mattes'):
        """runs the default version of the ANTs registration routine, that is a Rigid transformation, an affine trans-
        formation and a symmetric normalisation (SyN). Further options available using the cmdline """

        # sys.stdout = open(log_filename, 'w+') # TODO: logging not working; Must be rewritten possibly using logging module?!?
        transform = self.cfg["preprocess"]["registration"]["registration_method"]
        registered_image = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                             type_of_transform=transform, grad_step=.1,
                                             aff_metric=metric,
                                             outprefix=inputfolder,
                                             initial_transform="[%s,%s,1]" % (sequence1, sequence2),
                                             verbose=self.verbose)
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
        # TODO: add option for winsorize and for use_histogram or similar and add custom_registration_file to settings
        with open(os.path.join(ROOTDIR, "utils",
                               self.cfg["preprocess"]["registration"]["custom_registration_file"]), "r") as f:
            for line in f:
                processed_args.append(line.strip())

        replace_items = [('filename_fixed', sequencename1), ('filename_moving', sequencename2),
                         ('fixed_ANTsimage_process', fixed_pointer), ('moving_ANTsimage_process', moving_pointer),
                         ('input_folder', inputfolder),
                         ('warpedmovout_process', wmo_pointer), ('warpedfixout_process', wfo_pointer)]

        for item, replacement in replace_items:
            processed_args = [re.sub(r'[*]' + item + r'[*]', replacement, x) for x in processed_args]

        #orig_stdout = sys.stdout
        #sys.stdout = open(log_filename, 'w')
        libfn = ants.utils.get_lib_fn("antsRegistration")
        libfn(processed_args)
        #sys.stdout = orig_stdout
        #sys.stdout.close()

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

    def check_for_normalisation(self, subjects, fixed_image):
        """check that normalisation to T1-template was performed before CT-registration in order to avoid redundancy"""
        from itertools import compress

        bc_prefix = self.cfg["preprocess"]["ANTsN4"]["prefix"] + 't1_' # normalisation always with respect to T1
        if not fixed_image:
            fixed_image = self.cfg["preprocess"]["normalisation"]["prefix"] + 'run1_' + bc_prefix

        included_sequences = [bc_prefix.upper(), '{}'.format(fixed_image.upper())]

        incomplete = [False] * len(subjects)
        for idx, subj in enumerate(subjects):
            allfiles_subj = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=[subj])

            files_exist = []
            [files_exist.extend([os.path.isfile(x[0])])
             for x in allfiles_subj for y in included_sequences
             if re.search(r'\w+(?!_).({}).'.format(y), x[0], re.IGNORECASE)
             and x[0].endswith('.nii')]

            if not all(files_exist):
                incomplete[idx] = True

        subjects2normalise = list(compress(subjects, incomplete))
        if subjects2normalise:
            print('Of {} subjects\'s T1-imaging, {} were not yet normalised to template. \nStarting with '
                  'normalisation for {}'.format(len(subjects), len(subjects2normalise)),
                  ', '.join(x for x in subjects2normalise))
            self.ANTsCoregisterMRI2template(subjects2normalise) # running normalisation with all identified subjects
        else:
            print('T1-sequences of {} subject(s) was/were already normalised, proceeding!'.format(len(subjects)))

    @staticmethod
    def inner_join(a, b):
        """adapted from: https://stackoverflow.com/questions/31887447/how-do-i-merge-two-lists-of-tuples-based-on-a-key"""
        #TODO: put this into helper functions
        L = a + b
        L.sort(key=itemgetter(1))  # sort by the first column
        for _, group in groupby(L, itemgetter(1)):
            row_a, row_b = next(group), next(group, None)
            if row_b is not None:  # join
                yield row_b[0:1] + row_a  # cut 1st column from 2nd row
