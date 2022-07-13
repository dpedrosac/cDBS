#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import multiprocessing as mp
import os
import re
import shutil
import time
from itertools import compress

import ants
import numpy as np

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration, Imaging, FileOperations


# TODO: 1. second run of ANTsCoRegisterCT2MRI not working
# TODO: 2. logging ANTs output during registration not working

class RegistrationANTs:
    """All necessary functions to register pre- and postoperative data as well as imaging to templates."""

    def __init__(self):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.verbose = True

    def CoregisterMRI2template(self, subjects):
        """Co-Registration of preoperative MRI to specific template"""

        print('\nStarting Co-Registration for {} subject(s)'.format(len(subjects)))
        all_files = FileOperations.get_filelist_as_tuple(inputdir=self.cfg['folders']['nifti'], subjects=subjects)
        sequences = self.cfg['preprocess']['normalisation']['sequences'].split(sep=',')  # sequences of interest
        template = glob.glob(os.path.join(ROOTDIR, 'ext', 'templates',
                                          self.cfg['preprocess']['normalisation']['template_image'] + '/*'))

        fileIDs = []
        for idx, seqs in enumerate(sequences):
            file_template = [x for x in template if
                             re.search(r'\w+.({}).'.format(seqs), x, re.IGNORECASE)]  # corresponding template
            if not file_template:
                Output.msg_box(text="No template found. Please ensure templates are installed at './ext/templates'",
                               title="Template missing!")  # TODO install templates by default!
                return

            regex_complete = '{}{}'.format(self.cfg['preprocess']['ANTsN4']['prefix'], seqs)
            files_subj = [x for x in all_files if x[0].endswith('.nii') and 'run' not in x[0] and
                          re.search(r'\w+(?!_).({}).'.format(regex_complete), x[0], re.IGNORECASE)]
            fileIDs.extend(tuple([(file_template[0], file_id, subj) for file_id, subj in files_subj]))

        if not fileIDs:
            Output.msg_box(text="No bias-corrected MRI found. Please double-check", title="Preprocessed MRI missing")
            return

        self.wrapper_multiprocessing(fileIDs, subjects, 'MRI')

    def CoregisterCT2MRI(self, subjects, input_folder, fixed_image='reg_run[0-9]_bc_t1'):
        """Co-registration of postoperative CT to preoperative MRI for further analyses in same space; before registration
        presence of registered MRI data is ensured to avoid redundancy"""

        print('\nStarting co-registration for {} subject(s)'.format(len(subjects)))
        allfiles = FileOperations.get_filelist_as_tuple(inputdir=input_folder, subjects=subjects)
        self.check_for_normalisation(subjects)

        regex_complete = ['CT_', '{}_'.format(fixed_image.upper())]
        included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

        file_ID_CT, file_ID_MRI = ([] for _ in range(2))
        [file_ID_CT.append(x) for x in allfiles if 'run' not in x[0] and
         re.search(r'\w+{}.'.format(included_sequences[0]), x[0], re.IGNORECASE) and x[0].endswith('.nii')]

        [file_ID_MRI.append(x) for x in allfiles
         # for simplicity written in a second line as regexp is slightly different
         if re.search(r'\w+(?!_).({}).'.format(included_sequences[1]), x[0], re.IGNORECASE) and x[0].endswith('.nii')]

        if not file_ID_MRI:
            Output.msg_box(text="Bias-corrected MRI not found!", title="Preprocessed MRI unavailable")
            return
        fileIDs = list(FileOperations.inner_join(file_ID_CT, file_ID_MRI))

        self.wrapper_multiprocessing(fileIDs, subjects, 'CT')

    # ====================    Multiprocessing functions for both MRI2template and CT2MRI  ====================
    def wrapper_multiprocessing(self, fileIDs, subjects, modality):
        """To avoid redundancy between CT and MR  registration to common space, this part is wrapped up here """

        working_dir = self.cfg['folders']['nifti']
        prefix = '{}2template'.format(modality)

        # Start multiprocessing framework
        start_multi = time.time()
        status = mp.Queue()
        processes = [mp.Process(target=self.ANTsCoregisterMultiprocessing,
                                args=(filename_fixed, filename_moving, no_subj, os.path.join(working_dir, no_subj),
                                      'registration', prefix, status))
                     for filename_fixed, filename_moving, no_subj in fileIDs]

        for p in processes:
            p.start()

        while any([p.is_alive() for p in processes]):
            while not status.empty():
                filename_fixed, filename_moving, no_subj = status.get()
                print("\tRegistering {} (f) to {} (m), in ANTsPy\n".format(filename_fixed,
                                                                           os.path.split(filename_moving)[1], no_subj))
            time.sleep(0.1)

        for p in processes:
            p.join()

        # Functions creating/updating pipeline log, which documents all steps along with settings
        for subjID in subjects:
            files_processed = [(os.path.split(file_moving)[1], os.path.split(file_fixed)[1])
                               for file_fixed, file_moving, subj_no in fileIDs if subj_no == subjID]
            log_text = "{} successfully registered (@{}) to \n{}, \n\n Mean Duration per subject: {:.2f} " \
                       "secs".format(files_processed[0][0], time.strftime("%Y%m%d-%H%M%S"), files_processed[0][1],
                                     (time.time() - start_multi) / len(subjects))
            Output.logging_routine(text=Output.split_lines(log_text), cfg=self.cfg, subject=str(subjID),
                                   module='{}-Registration'.format(modality),
                                   opt=self.cfg['preprocess']['registration'], project="")

        print('\nIn total, a list of {} subject(s) was processed; {} registration took {:.2f}secs. '
              'overall'.format(len(subjects), modality, time.time() - start_multi))

    def ANTsCoregisterMultiprocessing(self, file_fixed, file_moving, subj, input_folder, flag, step, status, run=1):
        """Performs Co-Registration taking advantage of multicores, i.e. multiple subjects processed in parallel"""

        status.put(tuple([file_fixed, file_moving, subj]))
        prev_reg = glob.glob(os.path.join(input_folder + "/" + self.cfg['preprocess'][flag]['prefix'] + 'run*' +
                                          os.path.basename(file_moving)))
        if not prev_reg:
            print('\tNo previous registration found, starting with first run')
            filename_save = os.path.join(input_folder, self.cfg['preprocess'][flag]['prefix'] + 'run' +
                                         str(run) + '_' + os.path.basename(file_moving))
        elif re.search(r'\w+{}.'.format('CT_'), file_moving, re.IGNORECASE) and file_moving.endswith('.nii'):
            print('\tNo second run for CT-MRI registrations possible.')
            return
        else:
            allruns = [re.search(r'\w+(run)([\d.]+)', x).group(2) for x in prev_reg]
            lastrun = int(sorted(allruns)[-1])
            file_moving = os.path.join(input_folder, self.cfg["preprocess"]["registration"]["prefix"] + 'run' +
                                       str(lastrun) + '_' + os.path.basename(file_moving))  # update for second run
            run = lastrun + 1
            n4prefix = self.cfg['preprocess']['ANTsN4']['prefix']
            basename = '{}{}'.format(n4prefix, file_moving.split(n4prefix)[1])
            filename_save = os.path.join(input_folder, self.cfg['preprocess'][flag]['prefix'] + 'run' +
                                         str(run) + '_' + basename)

        print(filename_save)
        log_filename = os.path.join(ROOTDIR, 'logs', "log_Registration_using_ANTs_{}_run_{}_".format(subj, str(run)) +
                                    time.strftime("%Y%m%d-%H%M%S") + '.txt')

        imaging = dict()
        for idx, file_id in enumerate([file_fixed, file_moving]):
            sequence = ants.image_read(file_id)  # load data and resample images if necessary
            imaging[idx] = Imaging.resampleANTs(mm_spacing=self.cfg['preprocess']['registration']['resample_spacing'],
                                                ANTsImageObject=sequence, file_id=file_id,
                                                method=int(self.cfg['preprocess']['registration']['resample_method']))

        if run == 1:
            metric = self.cfg['preprocess']['registration']['metric'][0]
        else:
            self.cfg['preprocess']['registration'][
                'default_registration'] = 'yes'  # TODO: must be changed if non-default works
            metric = self.cfg['preprocess']['registration']['metric'][0]

        if self.cfg['preprocess']['registration']['default_registration'] == 'yes':
            registered_images = self.default_registration(imaging, file_fixed, file_moving, log_filename, metric=metric,
                                                          step=step)
        else:
            registered_images = self.custom_registration(imaging, file_fixed, file_moving,
                                                         input_folder + '/', log_filename, run)

        key2rename = {'fwdtransforms': ['{}_0GenericAffineRegistration.mat'.format(step), 1],
                      'invtransforms': ['{}_1InvWarpMatrix.mat'.format(step), 0]}
        for key, value in key2rename.items():
            Transform = ants.read_transform(registered_images[key][value[1]])
            ants.write_transform(Transform, os.path.join(input_folder, value[0]))

        ants.image_write(registered_images['warpedmovout'], filename=filename_save)
        FileOperations.create_folder(os.path.join(input_folder, "debug"))

        if run > 1:  # Previous registrations are moved to debug-folder
            prev_reg = ''.join(prev_reg) if type(prev_reg) == list else prev_reg
            filename_dest = re.sub(r'({}run[0-9]_)+({})'.format(self.cfg['preprocess']['registration']['prefix'],
                                                                self.cfg['preprocess']['ANTsN4']['prefix']),
                                   '{}RUNPREV{}_{}'.format(self.cfg['preprocess']['registration']['prefix'], lastrun,
                                                           self.cfg['preprocess']['ANTsN4']['prefix']),
                                   os.path.basename(prev_reg))
            shutil.move(prev_reg, os.path.join(input_folder, 'debug', filename_dest))

        create_mask = True
        if create_mask and 't1' in file_moving and not os.path.isfile(os.path.join(input_folder, 'brainmask_T1.nii')):
            Imaging.create_brainmask(input_folder, subj=subj, registered_images=registered_images['warpedmovout'])

    def default_registration(self, image_to_process, sequence1, sequence2, log_filename, metric='mattes', step='MRI'):
        """default version 'ANTs registration' routine, i.e. a Rigid transformation -> affine transformation ->
        symmetric normalisation (SyN). Further options available using the cmdline """
        # log_file = open(log_filename, 'w+') #TODO: logging not yet implemented
        transform = "[%s,%s,1]" % (sequence1, sequence2) if step == 'CT' else "[%s,%s,1]" % (
        sequence1, sequence2)  # "identity"
        registered = ants.registration(fixed=image_to_process[0], moving=image_to_process[1],
                                       type_of_transform=self.cfg['preprocess']['registration']['registration_method'],
                                       grad_step=.1, aff_metric=metric,  # outprefix=inputfolder,
                                       initial_transform=transform,
                                       verbose=self.verbose)
        return registered

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

        warpedmovout, warpedfixout = image_to_process[0].clone(), image_to_process[1].clone()

        fixed_pointer = ants.utils.get_pointer_string(image_to_process[0])
        moving_pointer = ants.utils.get_pointer_string(image_to_process[1])
        wfo_pointer = ants.utils.get_pointer_string(warpedfixout)
        wmo_pointer = ants.utils.get_pointer_string(warpedmovout)

        processed_args = []
        custom_registration = os.path.join(ROOTDIR, "utils",
                                           self.cfg['preprocess']['registration']['custom_registration_file'])
        # TODO: add option for winsorize and for use_histogram or similar and add custom_registration_file to settings
        with open(custom_registration, "r") as f:
            for line in f:
                processed_args.append(line.strip())

        replace_items = [('filename_fixed', sequencename1), ('filename_moving', sequencename2),
                         ('fixed_ANTsimage_process', fixed_pointer), ('moving_ANTsimage_process', moving_pointer),
                         ('input_folder', inputfolder),
                         ('warpedmovout_process', wmo_pointer), ('warpedfixout_process', wfo_pointer)]

        for item, replacement in replace_items:
            processed_args = [re.sub(r'[*]' + item + r'[*]', replacement, x) for x in processed_args]

        # orig_stdout = sys.stdout
        # sys.stdout = open(log_filename, 'w')
        libfn = ants.utils.get_lib_fn("antsRegistration")
        libfn(processed_args)
        # sys.stdout = orig_stdout
        # sys.stdout.close()

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

    def check_for_normalisation(self, subjects, fixed_image=''):
        """check available normalisation/registration to T1-template before CT-registration to avoid redundancy"""

        incomplete = [False] * len(subjects)
        for idx, subj in enumerate(subjects):
            inputfolder = os.path.join(self.cfg['folders']['nifti'], subj)
            prev_reg = glob.glob(os.path.join(inputfolder + "/" +
                                              self.cfg['preprocess']['registration']['prefix'] + 'run*'))
            incomplete[idx] = True if not prev_reg else False

        subjects2normalise = list(compress(subjects, incomplete))
        if subjects2normalise:
            print('Of {} subjects\' T1-imaging, {} was not yet registered to template. \nStarting normalisation '
                  'for {}'.format(len(subjects), len(subjects2normalise), ', '.join(x for x in subjects2normalise)))
            self.CoregisterMRI2template(subjects2normalise)  # running normalisation with all identified subjects
        else:
            print('\tT1-sequences of {} subject(s) was/were already normalised, proceeding!'.format(len(subjects)))
