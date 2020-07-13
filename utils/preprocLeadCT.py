#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import ants
import scipy
import math
import utils.HelperFunctions as HF
import multiprocessing as mp
import glob
from sklearn.decomposition import PCA
import numpy as np
from skimage.measure import regionprops, label
from scipy import ndimage
from dependencies import ROOTDIR

# constants to use throughout the script, should be included in the yaml configuration file
METAL_THRESHOLD = 800
LAMBDA_1 = 25

class LeadWorks:
    """this class contains all functions needed to detect leads and estimate their rotation angle. All scripts are based
    on the PACER algorithm (https://github.com/adhusch/PaCER)"""

    def __init__(self):
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)
        self.verbose = 0
        self.debug=False

    def LeadDetection(self, subjects, inputfolder=''):
        """creates a mask with information on leads according to a threshhold"""

        if not inputfolder:
            inputfolder = self.cfg["folders"]["nifti"]

        print('\nLead detection of {} subject(s)'.format(len(subjects)))
        allfiles = HF.get_filelist_as_tuple(inputdir=inputfolder, subjects=subjects)
        regex_complete = "reg_" + 'run'

        fileID = []
        [fileID.extend(x) for x in allfiles
         if re.search(r'\w.({}[0-9]).'.format(regex_complete), x[0], re.IGNORECASE)
         and x[0].endswith('.nii') and 'CT' in x[0]]

        fileID_brainmask = []
        [fileID_brainmask.extend(x) for x in allfiles
         if re.search(r'\w.(brainmask_).', x[0], re.IGNORECASE) and x[0].endswith('.nii')]

        if not fileID_brainmask:
            # TODO: here there should be a function which includes the brainmask in case not available, for that purpose, function should be called separately
            pass

        # here some code is needed which ensures that only one and only one file was found
        originalCT = ants.image_read(fileID[0])
        brainMask = ants.image_read(fileID_brainmask[0])

        if max(originalCT.spacing) > 1:
            print('Slice thickness > 1 mm! Independent contact detection most likely impossible. '
                  'Using contactAreaCenter based method.')
            flag = 'contactAreaCenter'
        elif max(originalCT.spacing) > .7:
            print('Slice thickness > 0.7 mm! reliable contact detection not guaranteed. However, for certain '
                  'electrode types with large contacts, it might work.')

        min_orig, max_orig = originalCT.min(), originalCT.max()
        rescaleCT = originalCT # Not needed necessarily

        # Start with electrode estimation
        leadPointcloudStruct = self.electrodeEstimation(fileID, rescaleCT, brainMask=brainMask, flag='')

        # Continue with lead processing


    def electrodeEstimation(self, fileID, CTimaging, brainMask, flag, threshold=''):
        """ estimates the electrode mask according to
        https://github.com/adhusch/PaCER/blob/master/src/Functions/extractElectrodePointclouds.m"""

        if CTimaging.dimension !=3:
            print('Something went wrong during CT-preprocessing. There are not three dimensions, please double-check')
            return

        if not threshold:
            threshold = METAL_THRESHOLD
            print('Routine estimation of number and location of DBS-leads @ standard threshold ({})'.format(threshold))
        else:
            print('Estimation of number and location of DBS-leads \w modified threshold ({})'.format(threshold))

        print('Thresholding {}: {} for content with HU > {}'.format(fileID[1], os.path.split(fileID[0])[1], threshold))
        if not self.debug: # the way Pacer estimates the mask, with an erosion of the boolean
            brainMaskmod = np.zeros(shape=CTimaging.shape, dtype=bool)
            brainMaskmod[brainMask.abs() > .97] = True
        else: # code compatible with debug from PaCER version in Lead-DBS package
            sphere_test = np.array(self.sphere(math.ceil(3 / max(CTimaging.spacing))))
            brainMaskmod = np.zeros(shape=CTimaging.shape, dtype=bool)
            brainMaskmod[brainMask.abs() > .5] = True
            brainMaskmod = ndimage.binary_erosion(brainMaskmod, structure=sphere_test).astype(bool) # erosion not necessary due to probabilistic maps and the possibility to change the threshold

        CTImagingData = CTimaging.numpy()
        CTImagingData[~brainMaskmod] = np.nan
        threshold_indices = np.zeros(shape=CTimaging.shape, dtype=bool)
        threshold_indices[CTImagingData > threshold] = True

        # largest connected components of metal inside of brain represents the electrodes
        cc = self.connected_objects(threshold_indices, connectivity=26)
        print('{} potential metal components were detected within the brain'.format(np.max(cc)))

        ccProps = regionprops(label_image=cc, intensity_image=None, cache=True,
                              coordinates=None)

        minVoxelNumber = (1.2 * 1.27 / 2) ** 2 * math.pi * 40 / np.prod(CTimaging.spacing) # according to PACER script, 40 mm in the brain and  20 % partial voluming; where are these values from
        maxVoxelNumber = (3 * 1.27 / 2) ** 2 * math.pi * 80 / np.prod(CTimaging.spacing)   # assuming 80 mm in brain and 300 % partial voluming

        areas = []
        [areas.append(a) for a in ccProps if a.area >= minVoxelNumber and a.area <= maxVoxelNumber]

        # TODO some feedback is needed here, saying how many artifacts/leads were detected

        leadPointcloudStruct = self.electrode_count(fileID, CTimaging, brainMask, threshold, areas)
        for leadPoints in leadPointcloudStruct:
            initialPoly, tPerMm, skeleton, totalLengthMm = self.electrodePointCloudModelEstimate(leadPoints)
            elecModels, intensityProfiles, skelSkelmms = \
                self.refitElec(self, initialPoly, leadPoints["points"], leadPoints["pixelValues"])
            #TODO: here matrices MUST be appended

    ## FIRST PART, detect leads
    def electrode_count(self, fileID, CTimaging, brainMask, threshold, areas):
        """estimate the number of electrodes found using the regionprops routine """

        if self.debug:
            # figure, scatterMatrix3(ccProps(1).PixelList)
            pass

        detected_leads = []
        for i, comp in enumerate(areas):
            pca = PCA()
            X = np.multiply(comp.coords, np.tile(list(CTimaging.spacing), (len(comp.coords),1))) #TODO: Is that equivalent?!? Especially the separate parts comp.coords and np.tile ...
            n_samples = X.shape[0]
            X_transformed = pca.fit_transform(X)
            X_centered = X - np.mean(X, axis = 0) # Is this necessary?
            cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
            latent = pca.explained_variance_
            if self.debug: # sanity check
                for latent_test, eigenvector in zip(latent, pca.components_):
                    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

            if len(latent) < 3:
                pass
                # return
            latent = np.sqrt(latent) * 2
            lowerAxesLength = sorted(latent[1:]) # TODO: Does this fit wit the PaCER algorithm?
            if (latent[0] > LAMBDA_1 and latent[0]/np.mean(latent[1:]) > 10
                    and lowerAxesLength[1] / (lowerAxesLength[0] + .001) < 8):
                detected_leads.append(comp)

        if not detected_leads:
            if threshold < 3000:
                print('Trying a higher threshold to ensure leads are detected.')
                leadPointcloudStruct, brainMask = self.electrodeEstimation(fileID, CTimaging, brainMask, threshold * 1.2)
                return leadPointcloudStruct, brainMask
            else:
                raise Exception('Even with higher thresholds, no leads could be detected. Double-check input !!')

        #transforms_raw = ants.read_transform("/media/storage/cDBS/data/NIFTI/subj1/0GenericAffineRegistration.mat")
        #transformation_matrix_ants = ants.apply_transforms_to_points(dim=3, points=detected_leads[1]["coords"],
        #                                                             transformlist=transforms_raw.parameters)
        #transformation_matrix_ants = ants.apply_ants_transform_to_point("/media/storage/cDBS/data/NIFTI/subj1/0GenericAffineRegistration.mat", detected_leads[1]["coords"])
        transformation_matrix = self.get_transformation_matrix(inputfolder=os.path.split(fileID[0])[0], inputfile='')
        transformation_matrix = np.multiply(np.eye(3), [round(f,1) for f in CTimaging.spacing])

        items = ["pixelList", "elecMask", "points", "pixelValues"]
        CTimagingData = CTimaging.numpy()
        leadPointcloudStruct = []
        for i, leadID in enumerate(detected_leads):
            # the transformations in the PaCer script are mostly skipped as all data is in the correct space and
            # transformations may be achieved using the matrix generated using the transform matrices
            leadPointcloudStruct.append({k: [] for k in items})
            pixelList = leadID["coords"]
            pointCloudMm = pixelList @ abs(transformation_matrix[:3, :3])
            pixelValues = [CTimagingData[tuple(pixelList[i])] for i, k in enumerate(pixelList)]
            elecMask_temp = np.zeros(shape=CTimaging.shape)
            for x,y,z in pixelList:
                elecMask_temp[x,y,z] = 1

            leadPointcloudStruct[i]["pixelList"] = pixelList
            leadPointcloudStruct[i]["points"] = pointCloudMm
            leadPointcloudStruct[i]["pixelValues"] = np.array(pixelValues)
            leadPointcloudStruct[i]["elecMask"] = elecMask_temp

            filename_elecMask = os.path.join(os.path.split(fileID[0])[0], 'elecMask_no' + str(i) + '.nii')
            ants.image_write(image=CTimaging.new_image_like(elecMask_temp), filename=filename_elecMask)

        return leadPointcloudStruct

    ## SECOND PART, estimate point cloud
    def electrodePointCloudModelEstimate(self, leadPoints, USE_REF_WEIGHTING=True):
        """ """
        POLYNOMIAL_ORDER = 8
        tol = 0

        zPlanes = np.unique(leadPoints["points"][:,-1])

        if not len(zPlanes) < leadPoints["points"].shape[0]:
            print('WARNING: CT planes in Z-direction not perfectly aligned; trying with tolerance')
            tol = .1
            zPlanes[~(np.triu(np.abs(zPlanes[:,None] - zPlanes) <= tol,1)).any(0)] #doesn't work!!!

        if not len(zPlanes) < leadPoints["points"].shape[0]:
            raise Exception('Somethings is wrong with the CT imaging')

        skeleton, sumInPlane = [[] * i for i in range(2)]
        for zplaneID in zPlanes:
            idx_zplane = np.where(abs(leadPoints["points"][:,-1] - zplaneID) <= tol)
            inPlanePoints = leadPoints["points"][idx_zplane, :]
            if USE_REF_WEIGHTING:
                inPlaneIntensities = leadPoints["pixelValues"][idx_zplane]
                skeleton.append(np.squeeze(inPlanePoints).T @ (inPlaneIntensities / np.sum(inPlaneIntensities)))
                sumInPlane.append(np.sum(inPlaneIntensities))
            else:
                skeleton.append = np.mean(inPlanePoints)

        skeleton = np.array(skeleton)
        sumInPlane = np.array(sumInPlane)

        # Filter skeleton for valid points
        filter = sumInPlane < np.median(sumInPlane) /1.5
        if sum(filter) > 0:
            print('Applied axial skeleton filter due to low intensity planes')
            skeleton = np.squeeze(skeleton[np.where(~filter) ,:])

        if all(skeleton[1,:] == np.zeros(3)):
            raise Exception('Empty skeleton. Was CT-imaging acquired in axial flow?')

        # Approximate parameterized polynomial([x y z] = f(t))
        if len(skeleton) < POLYNOMIAL_ORDER + 1:
            print('ElectrodePointCloudModelEstimate: less data points {} than internal poly-degree ({}). '
                  'Lowering degree but take care'.format(str(len(skeleton), str(POLYNOMIAL_ORDER))))
            POLYNOMIAL_ORDER = len(skeleton) - 1

        r3polynomial, tPerMm = self.fitParamPolytoSkeleton(skeleton, degree=POLYNOMIAL_ORDER)
        totalLengthMm = self.polyArcLength3(polyCoeff=r3polynomial, lowerLimit=[0], upperLimit=[1])

        return r3polynomial, tPerMm, skeleton, totalLengthMm

    def refitElec(self, initialPoly, pointCloud, voxelValues, **kwargs):
        """"""
        from scipy.interpolate import griddata
        # General options required for the function TODO: change this to lowercase as these are no static/global variables
        XY_RESOLUTION = 0.1
        Z_RESOLUTION = 0.025
        LIMIT_CONTACT_SEARCH_MM = 20 # limit contact search to first mm
        FINAL_DEGREE = 3

        # Algorithm
        #interpolationF = scatteredInterpolant(pointCloudWorld, double(voxelValues), 'linear', 'none'); % TODO check cubic!
        #interpolationF = griddata((pointCloud[:,1], pointCloud[:,2], pointCloud[:,3]), voxelValues, method='linear')
        totalLengthMm = self.polyArcLength3(initialPoly)
        totalLengthMm = [float(i) for i in totalLengthMm]
        XGrid, YGrid = np.meshgrid(np.arange(start=-1.5, stop=1.6, step=XY_RESOLUTION),
                                   np.arange(start=-1.5, stop=1.6, step=XY_RESOLUTION))

        oneMmEqivStep = 1 / totalLengthMm[0]
        STEP_SIZE = Z_RESOLUTION * oneMmEqivStep
        interpolationF = scipy.interpolate.LinearNDInterpolator(points=pointCloud, values=voxelValues)
        skeleton2nd, _, _, _ = self.oor(initialPoly, step_size=STEP_SIZE, XGrid, YGrid, interpolationF)
        refittedR3Poly2nd, _ = self.fitParamPolyToSkeleton(skeleton2nd, degree=8);

        skeleton3rd, medIntensity, orthIntensVol, _, skelScaleMm = self.oor(refittedR3Poly2nd, STEP_SIZE, XGrid, YGrid,
                                                                            interpolationF)
        dat1 = self.polyArcLength3(initialPoly)
        dat2 = self.polyArcLength3(refittedR3Poly2nd)

        print("\n 1st pass electrode length within Brain Convex Hull {}mm".format(dat1))
        print("\n 2nd pass electrode length within Brain Convex Hull {}mm".format(dat2))

        #return refitReZeroedElecMod, filteredIntensity, skelScaleMm

    def fitParamPolytoSkeleton(self, skeleton, degree=3):
        """"""

        ## Determine(approx.) t(i.e.LHS of parameterized regressor); (arg length to t)
        diffVecs = np.diff(skeleton, axis=0)
        apprTotalLengthMm = 0
        deltas, cumLengthMm = [np.zeros(len(diffVecs)) * i for i in range(2)]

        for k in range(0, len(diffVecs)):
            deltas[k] = np.linalg.norm(diffVecs[k,:])
            cumLengthMm[k] = np.sum(deltas)
            apprTotalLengthMm = apprTotalLengthMm + deltas[k]

        avgTperMm = 1 / apprTotalLengthMm
        t = np.append(0, np.divide(cumLengthMm, apprTotalLengthMm))# now range

        # Design matrix e.g.T = [t. ^ 4 t. ^ 3 t. ^ 2 t ones(length(t), 1)].'
        T = np.ones(shape=(len(t), degree+1))
        iter = -1
        for k in range(degree, 0, -1):
            iter += 1
            T[:,iter] = t ** k
        T = T.T

        # display(['Optimal AIC at degree ' num2str(polydeg(t, skeleton(:, 1))) ' for first (x) direction']);
        # display(['Optimal AIC at degree ' num2str(polydeg(t, skeleton(:, 2))) ' for second (y) direction']);
        # display(['Optimal AIC at degree ' num2str(polydeg(t, skeleton(:, 3))) ' for third (z) direction']);

        r3polynomial = np.linalg.lstsq(T.T, skeleton, rcond=None) # = skeleton' * pinv(T))' =(skeleton' * T'*inv
        # Asserts
        fittingErrs = np.sqrt(np.sum((r3polynomial[0].T @ T - skeleton.T) ** 2, axis=0))
        meanFittingError = np.mean(fittingErrs, axis=0)
        stdFittingError = np.std(fittingErrs, axis=0)
        maxFittingError = np.max(fittingErrs, axis=0)

        print('Max off-model: {:.4}, Mean off-model: {:.4}\n'.format(maxFittingError, meanFittingError))

        if (maxFittingError > 0.35 and maxFittingError > (meanFittingError + 3 * stdFittingError)):
            print('Check for outliers/make sure that the polynomial degree choosen is appropriate.\n In most cases this should be fine.\n')

        if self.debug:
            pass
        #    figure, plot(sum((r3polynomial' * T - skeleton').^ 2))

        return r3polynomial[0], avgTperMm

    def oor(self, r3Poly, step_size, xGrid, yGrid, interpolationF):
        """"""
        SND_THRESH = 1500
        arcLength = self.polyArcLength3(r3Poly);
        oneMmEqivStep = 1 / arcLength[0]
        lookahead = 3 * oneMmEqivStep
        regX, regY, regZ = r3Poly.T
        evalAtT = np.arange(start=-lookahead, stop=1, step=step_size)
        orthogonalSamplePoints, improvedSkeleton, avgIntensity, medIntensity, sumIntensity = [[] * i for i in range(5)]
        evalAtT = np.arange(start=-lookahead, stop=1, step=step_size)
        orthSamplePointsVol = np.zeros(r3Poly.shape[1]* len(xGrid) ** 2 * len(evalAtT)).reshape(r3Poly.shape[1],
                                                                                         len(xGrid) ** 2, len(evalAtT))
        orthIntensVol = np.zeros(xGrid.shape[0] * xGrid.shape[1] * len(evalAtT)).reshape(xGrid.shape[0], xGrid.shape[1],
                                                                                         len(evalAtT))

        for ind, evalAt in enumerate(evalAtT):
            # TODO: include a progressbar here
            x_d = np.polyval(np.polyder((regX)), evalAt)
            y_d = np.polyval(np.polyder((regY)), evalAt)
            z_d = np.polyval(np.polyder((regZ)), evalAt)
            direction = [x_d, y_d, z_d]
            currentPoint = np.polyval(r3Poly, evalAt)
            directionNorm = direction / np.linalg.norm(direction)

            ortho1 = np.cross(directionNorm, [0, 1, 0])
            ortho1 = ortho1 / np.linalg.norm(ortho1)
            ortho2 = np.cross(directionNorm, ortho1)
            ortho2 = ortho2 / np.linalg.norm(ortho2)

            orthogonalSamplePoints  = np.add(currentPoint[:,None],
                                             (np.dot(ortho1[:, None], np.ndarray.flatten(xGrid, order='F')[None, :]) + \
                                              np.dot(ortho2[:, None], np.ndarray.flatten(yGrid, order='F')[None, :])))
            orthSamplePointsVol[:,:,ind] = orthogonalSamplePoints

            intensities = interpolationF.__call__(orthogonalSamplePoints.T)
            intensitiesNanZero = np.copy(intensities)
            intensitiesNanZero[np.isnan(intensitiesNanZero)] = 0
            intensitiesNanZero[intensities < SND_THRESH] = 0

            # determine new skel point after rethresholding
            skelPoint = orthogonalSamplePoints * intensitiesNanZero / sum(intensitiesNanZero)
            if np.any(np.isnan(skelPoint)): # TODO needs check with original Matlab Code in order to ensure validity
                evalAtT[ind] = np.nan
                continue
            else:
                avgIntensity.append(np.nanmean(intensities)) #avgIntensity[ind] = np.nanmean(intensities)
                sumIntensity.append(np.nansum(intensitiesNanZero)) # sumIntensity[ind] = np.nansum(intensitiesNanZero)
                medIntensity.append(np.nanmedian(intensities)) # medIntensity[ind] = np.nanmedian(intensities)

                improvedSkeleton.append(skelPoint) #improvedSkeleton[ind,:] = skelPoint #  # ok<AGROW>
                intensityMap = np.reshape(intensitiesNanZero, (xGrid.shape[0], xGrid.shape[1]))
                orthIntensVol[:,:, ind] = intensityMap

            lowerLimits = np.zeros(shape=(len(evalAtT[~np.isnan(evalAtT)])))
            upperLimits = evalAtT[~np.isnan(evalAtT)]
            lowerLimits[upperLimits < 0] = upperLimits[upperLimits < 0]
            upperLimits[upperLimits < 0] = 0
            skelScaleMm = self.polyArcLength3(r3Poly, lowerLimit=lowerLimits, upperLimit=upperLimits)
            skelScaleMm = np.array(skelScaleMm)
            skelScaleMm[lowerLimits < 0] = -skelScaleMm[lowerLimits < 0]

        return improvedSkeleton, medIntensity, orthIntensVol, orthSamplePointsVol, skelScaleMm

    @staticmethod
    def connected_objects(data_array, connectivity):
        """ function creating a list of objects that are connected and satisfy certain conditions. This aims at
        replacing Mathworks bwconncomp.m function https://www.mathworks.com/help/images/ref/bwconncomp.html"""
        import cc3d

        labels_out = cc3d.connected_components(np.array(data_array), connectivity=connectivity)
        return labels_out

    @staticmethod
    def sphere(diameter):
        """function defining binary matrix which represents a 3D sphere which may be used as structuring element"""

        struct = np.zeros((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
        x, y, z = np.indices((2 * diameter + 1, 2 * diameter + 1, 2 * diameter + 1))
        mask = (x - diameter) ** 2 + (y - diameter) ** 2 + (z - diameter) ** 2 <= diameter ** 2
        struct[mask] = 1

        return struct.astype(np.bool)

    @staticmethod
    def get_transformation_matrix(inputfolder='', inputfile=''):
        """for reproducing the PaCER code this is necessary in order to get the registration done, taken from
        """
        from scipy.io import loadmat

        if not inputfile:
            # write a function looking for the affine...mat file
            inputfile = "/media/storage/cDBS/data/NIFTI/subj1/0GenericAffineRegistration.mat"

        # TODO: This unpythonic piece of code is a disaster in so many ways!!
        transformations = loadmat(inputfile)
        m_matrix = transformations["AffineTransform_float_3_3"][0:9].tolist()
        m_translation = transformations["AffineTransform_float_3_3"][9:12]
        m_fixed = transformations["fixed"]
        transformation_matrix = np.reshape(transformations["AffineTransform_float_3_3"][0:9], (3,3))
        transformation_matrix = np.append(transformation_matrix, m_translation, axis=1)
        transformation_matrix = np.append(transformation_matrix, np.array([[0,0,0,1]]), axis=0)

        mOffset = []
        for k in range(3):
            mOffset.append(np.array(m_translation[k] + m_fixed[k]).tolist())
            for j in range(3):
                mOffset[k] = np.array(mOffset[k] - (transformation_matrix[k, j] * m_fixed[j])).tolist()

        mOffset = list(np.array(mOffset).flat)
        transformation_matrix[0:3,3] = mOffset
        transformation_matrix = np.linalg.inv(transformation_matrix)

        return transformation_matrix

    @staticmethod
    def polyArcLength3(polyCoeff, lowerLimit=[0], upperLimit=[1]):
        """The arc length is defined as the integral of the norm of the derivatives of the parameterized equations.
        #arcLength(i) = integral(f, lowerLimit(i), upperLimit(i)); """
        from scipy.integrate import quad

        #epsilon = 0.001 # used to avoid numerical accuracy problems in assertion
        # assert(all(lowerLimit(:) <= upperLimit(:) + epsilon));

        # regX, regY, regZ = polyCoeff.T
        regX, regY, regZ = polyCoeff[:,0],  polyCoeff[:,1],  polyCoeff[:,2]
        x_d = np.polyder(regX) # TODO: there must be a more elegant way to define these
        y_d = np.polyder(regY)
        z_d = np.polyder(regZ)
        arcLength = np.zeros(len(lowerLimit))
        arcLength[:] = np.nan

        arcLength = []
        f_t = lambda x: np.sqrt(np.polyval(x_d, x) ** 2 + np.polyval(y_d, x) ** 2 + np.polyval(z_d, x) ** 2)
        for k in range(0,len(lowerLimit)):
            arcLength.append(quad(f_t, lowerLimit[k], upperLimit[k])[0])

        return arcLength
