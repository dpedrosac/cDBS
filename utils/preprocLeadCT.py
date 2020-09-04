#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle
import re
import time
import warnings

import ants
import numpy as np
import scipy
import scipy.io as spio
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.measure import regionprops
from sklearn.decomposition import PCA
from utils.HelperFunctions import Output, Configuration, FileOperations, Imaging, MatlabEquivalent

from dependencies import ROOTDIR


class LeadWorks:
    """Runs the algorithm for lead detection developed by Andreas Husch et al.
    see https://www.sciencedirect.com/science/article/pii/S2213158217302450?via%3Dihub
    It includes a three-step process: i) pre-processing, ii) and iii)"""

    def __init__(self):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.verbose = 0
        self.debug = False
        self.PacerEmulation = True

    def PaCER_script(self, subjects, inputfolder=''):
        """wrapper script for all steps included in the PaCER algorithm"""

        print("\nLead detection of {} subject(s)".format(len(subjects)))
        if not inputfolder:  # select default input folder
            inputfolder = self.cfg['folders']['nifti']

        # Look for data files containing CT imaging including the brainMask and load this into workspace
        available_files = FileOperations.get_filelist_as_tuple(inputdir=inputfolder, subjects=subjects)
        regex2lookfor = 'reg_' + 'run[0-9]', 'brainmask_'
        file_id_CTimaging = [file_tuple for file_tuple in available_files
                             if re.search(r'\w.({}).'.format(regex2lookfor[0]), file_tuple[0], re.IGNORECASE)
                             and file_tuple[0].endswith('.nii') and 'CT' in file_tuple[0]]

        file_id_brainMask = [file_tuple for file_tuple in available_files
                             if re.search(r'\w.({}).'.format(regex2lookfor[1]), file_tuple[0], re.IGNORECASE)
                             and file_tuple[0].endswith('.nii')]

        if any(t > 2 for t in [len(k) for k in file_id_CTimaging]):
            print("More than one files for imaging or brainmask available. Please double-check!")
            return

        if not file_id_brainMask:
            warnings.warn(message="\tNo brain mask was found, trying to obtain a mask using the ANTSpyNET routines")
            regex2lookforT1 = self.cfg['preprocess']['normalisation']['prefix'] + 'run'
            file_id_T1 = [file_tuple for file_tuple in available_files
                                 if re.search(r'\w.({}).'.format(regex2lookforT1), file_tuple[0], re.IGNORECASE)
                                 and 't1' in file_tuple[0] and file_tuple[0].endswith('.nii')]
            T1imaging = ants.image_read(file_id_T1[0][0])
            Imaging.create_brainmask(input_folder=inputfolder, registered_images=T1imaging)

        fileID = list(FileOperations.inner_join(file_id_brainMask, file_id_CTimaging))  # joins all information to one list

        metal_threshold = int(self.cfg['lead_detection']['PaCER']['metal_threshold'])
        elecModels, intensityProfiles, skelSkalms = self.electrodeEstimation(fileID[0], threshold=metal_threshold)

        filename_save = os.path.join(os.path.join(inputfolder, subjects[0]), 'elecModels_' + subjects[0] + '.pkl')
        with open(filename_save, "wb") as f:
            pickle.dump(elecModels, f)
            pickle.dump(intensityProfiles, f)
            pickle.dump(skelSkalms, f)

        print("Finished with lead detection!")
        # TODO: it does not return to the empty command line.

    def electrodeEstimation(self, fileID, threshold=1500):
        """ estimates the electrode mask with a threshold according to:
        https://github.com/adhusch/PaCER/blob/master/src/Functions/extractElectrodePointclouds.m"""

        CTimaging = ants.image_read(fileID[0])
        brainMask_prob = ants.image_read(fileID[1])  # probabilistic brainMask

        if CTimaging.dimension != 3:
            warnings.warn_explicit("\t Something went wrong during CT-preprocessing (ndim != 3)")
            return
        elif max(CTimaging.spacing) > 1:
            warnings.warn("\tSlice thickness > 1mm! Independent contact detection unlikely. Using 'contactAreaCenter'")
            self.cfg['lead_detection']['PaCER']['detection_method'] = 'contactAreaCenter'  # when spacing too big, this detection method is recommended
        elif max(CTimaging.spacing) > .7:
            warnings.warn("\tSlice thickness > .7mm! Reliable contact detection not guaranteed. For certain "
                          "lead types with large contacts, it may, however, work.")

        print("\tThresholding {}: {} for content w/ HU > {}".format(fileID[2], os.path.split(fileID[0])[1], threshold))
        brainMask = np.zeros(shape=CTimaging.shape, dtype=bool)
        if not self.PacerEmulation:
            brainMask[brainMask_prob.abs() > .99] = True
        else:  # code compatible with debug from PaCER version in Lead-DBS package
            sphere_test = np.array(Imaging.sphere(math.ceil(3 / max(CTimaging.spacing))))
            brainMask[brainMask_prob.abs() > .5] = True
            brainMask = ndimage.binary_erosion(brainMask, structure=sphere_test).astype(
                bool)  # erosion not necessary due to probabilistic maps and the possibility to change the threshold

        CTImagingData = CTimaging.numpy()
        CTImagingData[~brainMask] = np.nan
        threshold_indices = np.zeros(shape=CTimaging.shape, dtype=bool)
        threshold_indices[CTImagingData > threshold] = True  # creates a mask with ones wherever inside the brain

        # Largest connected components of metal inside brain represents the electrodes
        cc = self.connected_objects(threshold_indices, connectivity_values=26)
        print("\t{} potential metal components were detected within the brain.".format(np.max(cc)), end='')

        ccProps = regionprops(label_image=cc, intensity_image=None, cache=True, coordinates=None)
        minVoxelNumber = (1.2 * 1.27 / 2) ** 2 * math.pi * 40 / np.prod(
            CTimaging.spacing)  # according to PACER, 40 mm within brain & 20% partial voluming; source for values?
        maxVoxelNumber = (3 * 1.27 / 2) ** 2 * math.pi * 80 / np.prod(
            CTimaging.spacing)  # assuming 80 mm in brain and 300 % partial voluming

        areas = []  # Guessing areas of interest according to the minimum/maximum voxel number
        [areas.append(a) for a in ccProps if minVoxelNumber <= a.area <= maxVoxelNumber]
        print('Guessing {} of them being DBS-leads'.format(str(len(areas))))

        leadPointCloudStruct = self.identifyLeads(fileID, CTimaging, threshold, areas)  #
        elecModels, intensityProfiles, skelSkalms = [[] for _ in range(3)]
        for idx, leadPoints in enumerate(leadPointCloudStruct):
            print("\nAnalysing lead no {} with {} pixels".format("\u0332".join(str(idx + 1)),
                                                                   str(len(leadPoints['pixelList']))))
            initialPoly, tPerMm, skeleton, totalLengthMm = self.electrodePointCloudModelEstimate(leadPoints,
                                                                                                 CTimaging.spacing[2])
            mod, prof, skel = \
                self.refitElec(initialPoly, leadPoints["points"], leadPoints["pixelValues"], CTspace=CTimaging.spacing)
            elecModels.append(mod)
            intensityProfiles.append(prof)
            skelSkalms.append(skel)

        return elecModels, intensityProfiles, skelSkalms

    # ==============================    PREPROCESSING (1. step)  ==============================
    def identifyLeads(self, fileID, CTimaging, threshold, areas):
        """Estimate number of electrodes found using regionprops routine and generate a PointCloud for every lead.
         Details can be found in the book chapter by Husch et al (2015)"""

        detected_leads = []
        pca = PCA()
        for i, comp in enumerate(areas):
            X = np.multiply(comp.coords, np.tile(list(CTimaging.spacing), (len(comp.coords), 1)))
            n_samples = X.shape[0]
            X_transformed = pca.fit_transform(X)
            X_centered = X - np.mean(X, axis=0)  # Is this necessary?
            cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
            latent = pca.explained_variance_

            if self.debug:  # sanity check
                for latent_test, eigenvector in zip(latent, pca.components_):
                    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

            if len(latent) < 3:
                continue

            latent = np.sqrt(latent) * 2
            lowerAxesLength = sorted(latent[1:3])
            if (latent[0] >
                    float(self.cfg['lead_detection']['PaCER']['lambda']) and latent[0] / np.mean(latent[1:3]) > 10
                    and lowerAxesLength[1] / (lowerAxesLength[0] + .001) < 8):
                detected_leads.append(comp)

        if not detected_leads:
            while threshold < 3000:
                print("Trying a higher threshold to ensure leads are detected.")
                leadpoint_cloudstruct = self.electrodeEstimation(fileID, threshold * 1.2)
                return leadpoint_cloudstruct
            else:
                raise Exception("\t\tEven w/ thresholds around 3000 HU, no leads were detected. Double-check input!")

        # TODO include transformation matrix from file if selected in options
        transformation_matrix = np.multiply(np.eye(3), [round(f, 1) for f in
                                                        CTimaging.spacing])  # transformation only necessary if selected in options; otherwise all remains in "AN space"
        leadpoint_cloudstruct = []  # initialise variable in workspace
        items = ['pixelList', 'elecMask', 'points', 'pixelValues']
        CTimagingData = CTimaging.numpy()  # get the shape of CT imaging to later fill it with content

        for i, leadID in enumerate(detected_leads):
            leadpoint_cloudstruct.append({k: [] for k in items})
            pixelList = leadID['coords']
            leadpoint_cloudstruct[i]['pixelList'] = pixelList
            leadpoint_cloudstruct[i]['points'] = pixelList   @ abs(transformation_matrix[:3, :3])
            leadpoint_cloudstruct[i]['pixelValues'] = np.array([CTimagingData[tuple(pixelList[i])]
                                                                for i, k in enumerate(pixelList)])
            elecMask_temp = np.zeros(shape=CTimaging.shape)
            for x, y, z in pixelList:
                elecMask_temp[x, y, z] = 1

            leadpoint_cloudstruct[i]['elecMask'] = elecMask_temp
            filename_elecMask = os.path.join(os.path.split(fileID[0])[0], 'elecMask_no' + str(i) + '.nii')
            ants.image_write(image=CTimaging.new_image_like(elecMask_temp), filename=filename_elecMask)  # mask to NIFTI

        return leadpoint_cloudstruct

    # ==============================    PREPROCESSING (2. step)  ==============================
    def electrodePointCloudModelEstimate(self, leadPoints, spacing, USE_REF_WEIGHTING=True, tol=0):
        """creates a skeleton of the leadPoints at the centroid of it and returns the coefficients of a polynomial"""

        polynomial_ord = 8
        zPlanes = np.unique(leadPoints['points'][:, -1])

        if not len(zPlanes) < leadPoints['points'].shape[0]:
            warnings.warn("\tCT planes in z-direction not perfectly aligned; trying with tolerance")
            tol = .1
            zPlanes[~(np.triu(np.abs(zPlanes[:, None] - zPlanes) <= tol, 1)).any(0)]

        if not len(zPlanes) < leadPoints['points'].shape[0]:
            raise Exception('Somethings is wrong with the CT imaging, please double-check!')

        skeleton, sumInPlane = [[] * i for i in range(2)]
        for zplaneID in zPlanes:
            idx_zplane = np.where(abs(leadPoints['points'][:, -1] - zplaneID) <= tol)
            inPlanePoints = leadPoints['points'][idx_zplane, :]
            if USE_REF_WEIGHTING:
                inPlaneIntensities = leadPoints['pixelValues'][idx_zplane].astype('float32')  #
                # estimates slice wise centroid weighted by image intensity values (Husch et al. 2015, Section 2.1)
                skeleton.append(np.squeeze(inPlanePoints).T @ (inPlaneIntensities / np.sum(inPlaneIntensities)))
                sumInPlane.append(np.sum(inPlaneIntensities))
            else:
                skeleton.append = np.mean(inPlanePoints)

        skeleton, sumInPlane = np.array(skeleton), np.array(sumInPlane)
        skeleton_filter = sumInPlane < np.median(sumInPlane) / 1.5
        if sum(skeleton_filter) > 0:  # # filter skeleton for valid points
            print("\t\tApplied axial skeleton filter due to low intensity planes")
            skeleton = np.squeeze(skeleton[np.where(~skeleton_filter), :])

        if all(skeleton[1, :] == np.zeros(3)):
            raise Exception(
                "\t\t... empty skeleton. Was CT-imaging acquired in axial flow?")  # What on earth does that mean?

        # Approximate parameterized polynomial([x y z] = f(t))
        if len(skeleton) < polynomial_ord + 1:
            print("\t\tElectrodePointCloudModelEstimate: less data points {} than internal polynomial degree ({}). "
                  "caveat: lowering degree!".format(str(len(skeleton), str(polynomial_ord))))
            polynomial_ord = len(skeleton) - 1

        r3polynomial, tPerMm = self.fitParamPolytoSkeleton(skeleton, degree=polynomial_ord)  # Husch et al. 2015 eq.(1-4)
        totalLengthMm = self.polyArcLength3(polyCoeff=r3polynomial)

        return r3polynomial, tPerMm, skeleton, totalLengthMm

    def refitElec(self, initialPoly, pointCloud, voxelValues, CTspace, xy_resolution=.1, z_resolution=.025,
                  limit_contactsearch_mm=20, final_degree=1):
        """"""
        from scipy.interpolate import griddata

        totalLengthMm = self.polyArcLength3(initialPoly)
        totalLengthMm = [float(i) for i in totalLengthMm]
        XGrid, YGrid = np.meshgrid(np.arange(start=-1.5, stop=1.6, step=xy_resolution),
                                   np.arange(start=-1.5, stop=1.6, step=xy_resolution))

        oneMmEqivStep = 1 / totalLengthMm[0]
        STEP_SIZE = z_resolution * oneMmEqivStep
        interpolationF = scipy.interpolate.LinearNDInterpolator(points=pointCloud, values=voxelValues)

        print("\tFirst run:")
        skeleton2nd, _, _, _, _ = self.oor(initialPoly, STEP_SIZE, XGrid, YGrid, interpolationF)

        print("\tSecond run:")
        R3polynomial2nd, _ = self.fitParamPolytoSkeleton(np.array(skeleton2nd), degree=8)

        msg2plot = "\tRefitting parametrised polynomial to re-sampled data (2nd run)"
        skeleton3rd, medIntensity, orthIntensVol, _, skelScaleMm = self.oor(R3polynomial2nd, STEP_SIZE, XGrid, YGrid,
                                                                            interpolationF, run_information=msg2plot)
        dat1, dat2 = self.polyArcLength3(initialPoly), self.polyArcLength3(R3polynomial2nd)

        print("\n\t1st pass electrode length within Brain Convex Hull {:.4}mm".format(dat1[0]))
        print("\t2nd pass electrode length within Brain Convex Hull {:.4}mm".format(dat2[0]))

        filterWidth = (0.25 / z_resolution) + 1
        filteredIntensity = scipy.ndimage.filters.uniform_filter1d(medIntensity, size=int(filterWidth))
        filterIdxs = np.where(skelScaleMm <= limit_contactsearch_mm)

        peakLocs, peakWaveCenters, peakValues, threshIntensityProfile, threshold, contactAreaCenter, contactAreaWidth, \
        xrayMarkerAreaCenter, xrayMarkerAreaWidth = self.getIntensityPeaks(filteredIntensity, skelScaleMm,
                                                                           filterIdxs)

        detection_method = self.cfg['lead_detection']['PaCER']['detection_method']
        if detection_method == 'peakWaveCenters':
            contact_pos = peakWaveCenters
        else:
            contact_pos = peakLocs

        try:
            lead_information, dataModelPeakRMS = self.determineElectrodeType(contact_pos)
        except KeyError:
            print("Falling back to contact detection method: contactAreaCenter")
            detection_method == 'contactAreaCenter'

        if (len(contact_pos) < 4 or detection_method == 'contactAreaCenter'):
            if len(contact_pos) < 4:
                warnings.warn("\tCould not detect independent electrode contacts. Check image quality")
                return

            lead_geometries = spio.loadmat(os.path.join(ROOTDIR, 'ext', 'PaCER', 'electrodeGeometries.mat'),
                                           squeeze_me=True, simplify_cells=True)
            lead_geometries = lead_geometries['electrodeGeometries']
            lead_type = self.cfg['lead_detection']['PaCER']['lead_type']

            if lead_type == 'unknown':
                warnings.warn("\t\tNo lead specification provided, please set lead_type if possible. Meanwhile trying"
                              "to estimate type by width of contact area. Might be wrong, please double-check!")

                if contactAreaWidth < 10.5:
                    print("Assuming Boston Scientific Directional or Medtronic 3389. Setting former.")
                    lead_information = lead_geometries[2]
                else:
                    print("Assuming Medtronic 3387")
                    lead_information = lead_geometries[1]

            else:
                print("\t\tSetting user specified electrode type: {}".format(lead_type))
                try:
                    idx_leadInformation = [i for i, x in enumerate([lead_type == k['string']
                                                                    for k in lead_geometries]) if x][0]
                    lead_information = lead_geometries[idx_leadInformation]
                except IndexError:
                    warnings.warn("\t\tUnknown lead-type provided. Assuming default i.e. idx=[-1]")
                    lead_information = lead_geometries[-1]

            zeroT = self.invPolyArcLength3(R3polynomial2nd,
                                           np.array(contactAreaCenter - np.mean(
                                               lead_information['ringContactCentersMm'])))  # calibrate zero
        else:
            dataModelPeakRMS = 0
            if dataModelPeakRMS > 0.3:  # original comment: "the MAX deviation might be a better measure than the RMS?"
                print("\t\tSwitching to model-based contact positions because of high RMS "
                      "(Setting useDetectedContactPositions = 0).")
                useDetectedContactPositions = 0  # TODO what is the purpose of this
            zeroT = self.invPolyArcLength3(R3polynomial2nd,
                                           contact_pos[0] - lead_information['zeroToFirstPeakMm'])
            refittedContactDistances = contact_pos - (contact_pos[0] - lead_information['zeroToFirstPeakMm'])

        if final_degree == 1:
            elecEndT = self.invPolyArcLength3(R3polynomial2nd, np.array(limit_contactsearch_mm))
            spacing = np.linspace(start=zeroT, stop=elecEndT, num=math.floor(totalLengthMm[0] / xy_resolution))
            poly_coeffs = []
            for i, coords in enumerate(R3polynomial2nd.T):
                poly_coeffs.append(np.polyval(coords, spacing))

            refittedR3PolyTmp = self.fitParamPolytoSkeleton(np.array(poly_coeffs).T, final_degree)
            spacing = np.linspace(start=zeroT, stop=self.invPolyArcLength3(refittedR3PolyTmp[0],
                                                                           np.array(totalLengthMm))[0],
                                  num=math.floor(totalLengthMm[0] / xy_resolution))

            poly_coeffs = []
            for i, coords in enumerate(refittedR3PolyTmp[0].T):
                poly_coeffs.append(np.polyval(coords, spacing))

            refittedR3PolyReZeroed = self.fitParamPolytoSkeleton(np.array(poly_coeffs).T, final_degree)
        else:  # normal case
            spacing = np.linspace(start=zeroT, stop=1, num=math.floor(totalLengthMm[0] / xy_resolution))

            poly_coeffs = []
            for i, coords in enumerate(R3polynomial2nd.T):
                poly_coeffs.append(np.polyval(coords, spacing))

            refittedR3PolyReZeroed = self.fitParamPolyToSkeleton(np.array(poly_coeffs), final_degree)

        print("\t\t\tElectrode Length within Brain Convex Hull after contact detection and Zero-Point calibration: "
              "{:.5} mm".format(str(self.polyArcLength3(refittedR3PolyReZeroed[0], 0, 1)[0])))
        refitReZeroedElecMod = self.summarise_results(refittedR3PolyReZeroed[0], lead_information, lead_type, CTspace)

        return refitReZeroedElecMod, filteredIntensity, skelScaleMm

    def fitParamPolytoSkeleton(self, skeleton, degree=3):
        """This function models the lead in a parametrised way according to (sec 2.2) Husch et al. 2015"""

        diff_vector = np.diff(skeleton, axis=0)
        approxTotalLengthMm = 0  # approximated total length [in mm]
        deltas, cumLengthMm = [np.zeros(len(diff_vector)) * i for i in range(2)]

        for k in range(0, len(diff_vector)):
            deltas[k] = np.linalg.norm(diff_vector[k, :])  # according to eq(3) in Husch et al 2015,
            cumLengthMm[k] = np.sum(deltas)
            approxTotalLengthMm = approxTotalLengthMm + deltas[k]

        avgStepsPerMm = 1 / approxTotalLengthMm # average steps [per mm]
        t = np.append(0, np.divide(cumLengthMm, approxTotalLengthMm))  # 0 at start->len(t)=len(skel.), norm. to [0, 1]

        # Design matrix e.g.T = [t. ^ 4 t. ^ 3 t. ^ 2 t ones(length(t), 1)].'
        T = np.ones(shape=(len(t), degree + 1))  # for details cf. eq(4) Husch et al. 2015
        iter = -1
        for k in range(degree, 0, -1):
            iter += 1
            T[:, iter] = t ** k
        T = T.T

        r3polynomial, _, _, _ = np.linalg.lstsq(T.T, skeleton, rcond=None)  # OLS solution for linear regression with T as coeffs.
        fittingErrs = np.sqrt(np.sum((r3polynomial.T @ T - skeleton.T) ** 2, axis=0))
        meanFittingError = np.mean(fittingErrs, axis=0)
        stdFittingError = np.std(fittingErrs, axis=0)
        maxFittingError = np.max(fittingErrs, axis=0)

        print("\t\t\tMax off-model: {:.4}, Mean off-model: {:.4}\n".format(maxFittingError, meanFittingError))
        if maxFittingError > 0.35 and maxFittingError > (meanFittingError + 3 * stdFittingError):
            print("\t\tCheck for outliers/make sure chosen polynomial degree is appropriate.\n "
                  "\t\tIn most cases selection should be fine.\n")

        return r3polynomial, avgStepsPerMm

    def oor(self, r3Poly, step_size, xGrid, yGrid, interpolationF, run_information=''):
        """optimal oblique re-sampling; routine enabling automatic contact detection by creating perpendicular slices
        with respect to the lead """
        if not run_information:
            run_information = "\t\tEstimating oblique slices which are orthogonal to first-pass electrode"

        arcLength = self.polyArcLength3(r3Poly)
        oneMmEqivStep = 1 / arcLength[0]
        lookahead = 3 * oneMmEqivStep

        poly_coeffs = []
        for coords in r3Poly.T:
            poly_coeffs.append(np.polyder(coords))

        evalAtT = np.arange(start=-lookahead, stop=1, step=step_size)  # create samples of all available datapoints
        iters = len(evalAtT)
        orthogonalSamplePoints, improvedSkeleton, avgIntensity, medIntensity, sumIntensity = [[] * i for i in range(5)]
        orthSamplePointsVol = np.zeros(r3Poly.shape[1] * len(xGrid) ** 2 * len(evalAtT)).reshape(r3Poly.shape[1],
                                                                                                 len(xGrid) ** 2,
                                                                                                 len(evalAtT))
        orthIntensVol = np.zeros(xGrid.shape[0] * xGrid.shape[1] * len(evalAtT)).reshape(xGrid.shape[0], xGrid.shape[1],
                                                                                         len(evalAtT))
        print("{}".format(run_information))
        Output.printProgressBar(0, iters, prefix="\t\tProgress:", suffix='Complete', length=50)
        for ind, relative_location in enumerate(evalAtT):
            time.sleep(.01)
            Output.printProgressBar(ind + 1, iters, prefix="\t\tProgress:", suffix='Complete', length=50)
            direction = [np.polyval(x, relative_location) for x in poly_coeffs]
            currentPoint = np.polyval(r3Poly, relative_location)
            directionNorm = direction / np.linalg.norm(direction)

            ortho1 = np.cross(directionNorm, [0, 1, 0])
            ortho1 = ortho1 / np.linalg.norm(ortho1)
            ortho2 = np.cross(directionNorm, ortho1)
            ortho2 = ortho2 / np.linalg.norm(ortho2)

            orthogonalSamplePoints = np.add(currentPoint[:, None],
                                            (np.dot(ortho1[:, None], np.ndarray.flatten(xGrid, order='F')[None, :]) +
                                             np.dot(ortho2[:, None], np.ndarray.flatten(yGrid, order='F')[None, :])))
            orthSamplePointsVol[:, :, ind] = orthogonalSamplePoints

            intensities = interpolationF.__call__(orthogonalSamplePoints.T)
            intensitiesNanZero = np.where(np.isnan(np.copy(intensities)), 0, intensities)
            intensitiesNanZero[intensitiesNanZero < float(self.cfg['lead_detection']['PaCER']['snr_threshold'])] = 0

            with np.errstate(divide='ignore', invalid='ignore'):
                skelPoint = orthogonalSamplePoints @ intensitiesNanZero / sum(intensitiesNanZero)

            if np.any(np.isnan(skelPoint)):
                evalAtT = evalAtT[1:]
                # evalAtT[ind] = np.nan
                continue
            else:
                avgIntensity.append(np.nanmean(intensities))  # avgIntensity[ind] = np.nanmean(intensities)
                sumIntensity.append(np.nansum(intensitiesNanZero))  # sumIntensity[ind] = np.nansum(intensitiesNanZero)
                medIntensity.append(np.nanmedian(intensities))  # medIntensity[ind] = np.nanmedian(intensities)

                improvedSkeleton.append(list(skelPoint))
                orthIntensVol[:, :, ind] = np.reshape(intensitiesNanZero, (xGrid.shape[0], xGrid.shape[1],))

        lowerLimits = np.zeros(shape=(len(evalAtT[~np.isnan(evalAtT)])))
        upperLimits = evalAtT[~np.isnan(evalAtT)]
        lowerLimits[upperLimits < 0] = upperLimits[upperLimits < 0]
        upperLimits[upperLimits < 0] = 0
        skeletonScaleMm = self.polyArcLength3(r3Poly, lowerLimit=lowerLimits, upperLimit=upperLimits)
        skeletonScaleMm = np.array(skeletonScaleMm)
        skeletonScaleMm[lowerLimits < 0] = -skeletonScaleMm[lowerLimits < 0]

        return improvedSkeleton, medIntensity, orthIntensVol, orthSamplePointsVol, skeletonScaleMm

    def getIntensityPeaks(self, filteredIntensity, skelScaleMm, filterIdxs):
        """Detection of center-line pointcloud using intensity-weighted means (see Husch et al. 2018, sec 2.4)"""
        from scipy.signal import find_peaks
        from scipy.ndimage import label

        peaks, properties = find_peaks(filteredIntensity[filterIdxs],
                                       distance=1.4, height=1.1 * np.nanmean(filteredIntensity),
                                       prominence=.01 * np.nanmean(filteredIntensity))
        xrayMarkerAreaWidth, xrayMarkerAreaCenter = [[] * i for i in range(2)]
        try:
            threshold = min(filteredIntensity[peaks[0:4]]) - (min(properties['prominences'][0:4]) / 4)
            threshIntensityProfile = np.minimum(filteredIntensity, threshold)
            contactSampleLabels = label(~(threshIntensityProfile[filterIdxs] < threshold))
            values = MatlabEquivalent.accumarray(skelScaleMm[filterIdxs], contactSampleLabels[0])
            counts = np.bincount(contactSampleLabels[0])
            peakWaveCenters = values[1:5] / counts[1:5]  # index 0 is the "zero label"
        except:
            print("\tpeakWaveCenter detection failed. Returning peaksLocs in peakWaveCenters.")
            peakWaveCenters = skelScaleMm[peaks]
            threshIntensityProfile = filteredIntensity
            threshold = np.nan

        # Detect 'contact area' as fallback for very low SNR signals where no single contacts are visible
        thresholdArea = np.mean(filteredIntensity[filterIdxs])
        threshIntensityProfileArea = np.minimum(filteredIntensity, thresholdArea)
        contactSampleLabels = label(~(threshIntensityProfileArea[filterIdxs] < thresholdArea))
        values = MatlabEquivalent.accumarray(skelScaleMm[filterIdxs], contactSampleLabels[0])
        counts = np.bincount(contactSampleLabels[0])
        contactAreaCenter = values[1] / counts[
            1]  # index 0 is the "zero label", index 2( value 1) is the contact region, index 3 (value 2) might be an X - Ray obaque arker

        idxs = np.where(contactSampleLabels[0] + 1 == 2)
        contactAreaWidth = np.abs(skelScaleMm[idxs[0][0]] - skelScaleMm[idxs[0][-1]])
        if np.max(contactSampleLabels[0]) > 1:
            print(
                '\tMultiple metal areas found along electrode. Possibly an electrode type with addtional X-Ray marker!')
            xrayMarkerAreaCenter = values[2] / counts[
                2]  # index 1 is the "zero label", index 2  (value 1) is the contact region, index 3 (value 2) might be an X-Ray obaque arker
            idxs = np.where(contactSampleLabels[0] + 1 == 3)
            xrayMarkerAreaWidth = np.abs(skelScaleMm[idxs[0][0]] - skelScaleMm[idxs[0][-1]])

        if self.debug:
            plt.plot(skelScaleMm[filterIdxs], filteredIntensity[filterIdxs])
            plt.scatter(skelScaleMm[peaks], filteredIntensity[peaks], edgecolors='red')
            plt.plot(skelScaleMm[filterIdxs], threshIntensityProfileArea[filterIdxs])
            plt.scatter(peakWaveCenters, [threshold] * 4)
            plt.grid(color='grey', linestyle='-', linewidth=.25)

        return peaks, peakWaveCenters, properties, threshIntensityProfile, threshold, contactAreaCenter, \
               contactAreaWidth, xrayMarkerAreaCenter, xrayMarkerAreaWidth

    @staticmethod
    def connected_objects(data_array, connectivity_values=26):
        """ function creating a list of objects that are connected and satisfy certain conditions. This aims at
        replacing Mathworks bwconncomp.m function https://www.mathworks.com/help/images/ref/bwconncomp.html"""
        import cc3d
        labels_out = cc3d.connected_components(np.array(data_array), connectivity=connectivity_values)
        return labels_out

    # ==============================    local Helper functions   ==============================
    @staticmethod
    def polyArcLength3(polyCoeff, lowerLimit=0, upperLimit=1):  # equation (2) Husch et al. 2018
        """The arc length is defined as the integral of the norm of the derivatives of the parameterized equations.
        #arcLength(i) = integral(f, lowerLimit(i), upperLimit(i)); """
        from scipy.integrate import quad

        try:
            int(lowerLimit)
            lowerLimit = [lowerLimit]
            upperLimit = [upperLimit]
        except TypeError:
            epsilon = 0.001  # used to avoid numerical accuracy problems in assertion
            if not np.all(lowerLimit[:] <= upperLimit[:] + epsilon):
                raise('There is an accuracy problem here!')

        regX, regY, regZ = polyCoeff[:, 0], polyCoeff[:, 1], polyCoeff[:, 2]
        x_d, y_d, z_d = np.polyder(regX), np.polyder(regY), np.polyder(regZ)

        arcLength = []
        f_t = lambda x: np.sqrt(np.polyval(x_d, x) ** 2 + np.polyval(y_d, x) ** 2 + np.polyval(z_d, x) ** 2)
        for lowerlim, upperlim in zip(lowerLimit, upperLimit):
            arcLength.append(quad(f_t, lowerlim, upperlim)[0])

        return arcLength

    def invPolyArcLength3(self, polyCoeff, arcLength):  # eq. (3) in Husch et al. 2018
        """according to conditions resumed in paper, inverse of integral (arcLength) can be estimated as follows """

        fx_invpolyArc = lambda x, a, coeff: np.abs(a - self.polyArcLength3(coeff, [0], x))
        if len(arcLength.shape) != 0:  # bulky snipped of code ensuring that single float values are processed as well
            inv_arcLength = []
            for i, arc_lgth in enumerate(arcLength):
                inv_arcLength.append(scipy.optimize.fmin(func=fx_invpolyArc, x0=[0], args=(arc_lgth, polyCoeff,),
                                                         disp=0)[0])
        else:
            arcLength = arcLength[()]
            inv_arcLength = scipy.optimize.fmin(func=fx_invpolyArc, x0=[0], args=(arcLength, polyCoeff,), disp=0)[0]

        return inv_arcLength

    @staticmethod
    def determineElectrodeType(peakDistances):
        """determines the most suitable electrode type based on the euclidean distance between detected and specified
        peaks in the collected data. Data should should be provided in the form of the file ‘electrodeGeometries.mat’"""

        electrodeGeometries = spio.loadmat(os.path.join(ROOTDIR, 'ext', 'PaCER', 'electrodeGeometries.mat'),
                                           squeeze_me=True, simplify_cells=True)
        electrodeGeometries = electrodeGeometries['electrodeGeometries']

        distances, rms = [np.zeros(len(electrodeGeometries)) for _ in range(2)]

        for idx, geoms in enumerate(electrodeGeometries):
            try:
                distances[idx] = np.linalg.norm(np.diff(peakDistances) - geoms['diffsMm'])
                rms[idx] = np.sqrt(np.mean(np.diff(peakDistances) - geoms['diffsMm']) ** 2)
            except ValueError:
                distances[idx] = float('inf')
                rms[idx] = float('inf')

        if np.all(np.isinf(distances)):
            print("determineElectrodeType: Could NOT detect electrode type, thus contact detection might be flawed "
                  "(Low image resolution? Large slice thickness!?) Set electrode type manually to continue with data")
            elecStruct = electrodeGeometries[-1]

            return elecStruct, rms

        d = np.min(distances)
        idx = np.argmin(distances)
        rms = rms[np.where(distances == d)]
        print("\t\tdetermineElectrodeType: data to model peak/contact spacing RMS distance is {} mm".format(str(rms)))
        elecStruct = electrodeGeometries[idx]

        return elecStruct, rms

    def summarise_results(self, r3polynomial, lead_information, lead_type, CTspace):
        """function aiming at providing an overview of the results obtained"""

        items1 = 'lead_diameter', 'lead_color', 'active_contact_color', 'alpha', 'metal_color', \
                 'r3polynomial', 'referenceTrajectory', 'leadInfo', 'activeContact', 'detectedContactPosition', \
                 'useDetectedContactPosition', 'skeleton', 'contactPositions', 'getContactPositions3D', 'trajectory', \
                 'markers_head', 'markers_tail', 'normtraj_vector', 'orth', 'markers_x', 'markers_y', 'rotation', \
                 'manual_correction', 'first_run'
        refitReZeroedElecMod = {k: [] for k in items1}

        refitReZeroedElecMod['lead_information'] = lead_information
        refitReZeroedElecMod['lead_diameter'] = 1.27
        refitReZeroedElecMod['lead_color'] = 0.1171875, 0.5625, 1
        refitReZeroedElecMod['active_contact_color'] = 1, 0.83984375, 0
        refitReZeroedElecMod['alpha'] = .9
        refitReZeroedElecMod['metal_color'] = .75, .75, .75
        refitReZeroedElecMod['r3polynomial'] = r3polynomial
        refitReZeroedElecMod['activecontact'] = np.nan
        refitReZeroedElecMod['detectedContactPosition'] = []
        refitReZeroedElecMod['UseDetectedContactPosition'] = False
        refitReZeroedElecMod['skeleton'] = self.create_skeleton(r3polynomial)
        refitReZeroedElecMod['contactPositions'] = .75, 2.75, 4.75, 6.75
        refitReZeroedElecMod['apprTotalLengthMm'] = []
        refitReZeroedElecMod['activeContactPoint'] = []

        positions = self.invPolyArcLength3(r3polynomial, np.array(refitReZeroedElecMod['contactPositions']))
        poly_coeffs = []
        for i, coords in enumerate(r3polynomial.T):
            poly_coeffs.append(np.polyval(coords, positions))
        refitReZeroedElecMod['getContactPositions3D'] = np.concatenate((poly_coeffs,
                                                                        np.ones((1,4)))).T @ np.eye(4) / CTspace
        refitReZeroedElecMod['getContactPositions3D'] = refitReZeroedElecMod['getContactPositions3D'][:,:3]

        trajectory = []
        for dim in range(3):
            trajectory.append(np.linspace(start=refitReZeroedElecMod['getContactPositions3D'][0, dim],
                                          stop=refitReZeroedElecMod['getContactPositions3D'][0, dim] +
                                               10 * (refitReZeroedElecMod['getContactPositions3D'][0, dim] -
                                                     refitReZeroedElecMod['getContactPositions3D'][-1, dim]), num=20))  # TODO: differences between last contact and first are better suited for the trajectory
        refitReZeroedElecMod['trajectory'] = np.array(trajectory).T
        refitReZeroedElecMod['markers_head'] = refitReZeroedElecMod['getContactPositions3D'][0, :]
        refitReZeroedElecMod['markers_tail'] = refitReZeroedElecMod['getContactPositions3D'][3, :]

        refitReZeroedElecMod['normtraj_vector'] = (refitReZeroedElecMod['getContactPositions3D'][0, :] -
                                                   refitReZeroedElecMod['getContactPositions3D'][3, :]) / \
                                                  np.linalg.norm(refitReZeroedElecMod['getContactPositions3D'][0, :] -
                                                                 refitReZeroedElecMod['getContactPositions3D'][3, :])
        refitReZeroedElecMod['orth'] = np.multiply(self.null(refitReZeroedElecMod['normtraj_vector']),
                                                   (refitReZeroedElecMod['lead_diameter'] / 2))
        refitReZeroedElecMod['markers_x'] = refitReZeroedElecMod['getContactPositions3D'][0, :] + \
                                            refitReZeroedElecMod['orth'][:, 0]
        refitReZeroedElecMod['markers_y'] = refitReZeroedElecMod['getContactPositions3D'][0, :] + \
                                            refitReZeroedElecMod['orth'][:, 1]
        refitReZeroedElecMod['model'] = lead_type
        refitReZeroedElecMod['manual_correction'] = False
        refitReZeroedElecMod['first_run'] = False

        return refitReZeroedElecMod

    @staticmethod
    def create_skeleton(r3polynomial):
        evalAtT = np.arange(start=0, stop=1, step=1 / 1000)  # create samples of all available datapoints
        refittedSkeleton = [np.polyval(x, evalAtT) for x in r3polynomial.T]
        return np.array(refittedSkeleton).T

    @staticmethod
    def null(A, atol=1e-13, rtol=0):
        A = np.atleast_2d(A)
        u, s, vh = np.linalg.svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T

        return ns
