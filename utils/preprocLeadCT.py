#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import ants
import scipy
import math
import time
import utils.HelperFunctions as HF
import multiprocessing as mp
import glob
from sklearn.decomposition import PCA
import numpy as np
import shutil
from skimage.measure import regionprops
from scipy import ndimage
from itertools import groupby
from operator import itemgetter
from dependencies import ROOTDIR

# constants to use throughout the script
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
        #        allfiles = HF.get_filelist_as_tuple(inputdir=self.cfg["folders"]["nifti"], subjects=subjects)
        allfiles = HF.get_filelist_as_tuple(inputdir=inputfolder, subjects=subjects)
        #reg_run1_02_Spirale_Trauma_CCT_0.75_H30s_e1.nii
        #regex_complete = self.cfg["preprocess"]["registration"]["prefix"]  + '_run' # TODO: change for final version
        regex_complete = "reg_" + 'run'
        #included_sequences = [x for x in list(filter(re.compile(r"^(?!~).*").match, regex_complete))]

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

        # TODO: double-check with PACER algorithm if same CT is loaded (that is spacing, size, etc)
        if max(originalCT.spacing) > 1:
            print('Slice thickness > 1 mm! Independent contact detection most likely impossible. '
                  'Using contactAreaCenter based method.')
            flag = 'contactAreaCenter'
        elif max(originalCT.spacing) > .7:
            print('Slice thickness > 0.7 mm! reliable contact detection not guaranteed. However, for certain '
                  'electrode types with large contacts, it might work.')

        min_orig, max_orig = originalCT.min(), originalCT.max()
        rescaleValues = ants.contrib.RescaleIntensity(min_orig, max_orig)
        #rescaleValues = ants.contrib.RescaleIntensity(-1024, 4096)  # to avoid values <0 causing problems w/ log data; maybe wrong and should rather be in if else condition as in Pacer
        rescaleCT = rescaleValues.transform(originalCT)

        # Start with electrode estimation
        cc = self.electrodeEstimation(fileID, rescaleCT, brainMask=brainMask, flag='')

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

        #brainMask = np.zeros(CTimaging.shape) # TODO: create a true brainMask
        #brainMask[120:150, 100:300, 75:190] =1
        # TODO: double-check size/dimensions of brainMask with PACER
        print('Thresholding {}: {} for artifacts/content with HU > {}'.format(fileID[1], os.path.split(fileID[0])[1],
                                                                              threshold))
        sphere_test = np.array(self.sphere(math.ceil(3/max(CTimaging.spacing))))
        brainMasko = brainMask.abs() # TODO: bulky form of replacing probabilistic data. This requires an even more sophisticated way
        brainMasko[brainMasko>.5] = 1
        brainMasko[brainMasko < .5] = 0
        brainMasko = ndimage.binary_dilation(brainMasko, structure=sphere_test).astype(bool) # TODO: change this to one line command

        dataImage_array = CTimaging.numpy()
        dataImage_array[~brainMasko] = np.nan

        threshold_image = dataImage_array > threshold # TODO: Why is that necessary?

        # largest connected components of metal inside of brain represents the electrodes
        cc = self.connected_objects(threshold_image, connectivity=26)
        print('{} potential metal components were detected within the brain'.format(np.max(cc)))
        # replacement for regionprops == scikit-image?
        ccProps = regionprops(label_image=cc, intensity_image=None, cache=True,
                    coordinates=None)

        # select only areas that meet certain conditions:
        minVoxelNumber = (1.2 * 1.27 / 2) ** 2 * math.pi * 40 / np.prod(CTimaging.spacing) # TODO: is niiCT.voxsize really the same thing as CTimaging.spacing? # according to PACER script, 40 mm in the brain and  20 % partial voluming
        maxVoxelNumber = (3 * 1.27 / 2) ** 2 * math.pi * 80 / np.prod(CTimaging.spacing)   # TODO: Is that value identical to PACER?   # assuming 80 mm in brain and 300 % partial voluming

        areas = []
        [areas.append(a) for a in ccProps if a.area >= minVoxelNumber and a.area <= maxVoxelNumber]

        self.electrode_count(fileID, CTimaging, brainMask, threshold, areas)
        CTimaging = CTimaging.new_image_like(dataImage_array)


        return CTimaging

    def electrode_count(self, fileID, CTimaging, brainMask, threshold, areas):
        """estimate the number of electrodes found using the regionprops routine """

        if self.debug:
            # figure, scatterMatrix3(ccProps(1).PixelList)
            pass

        elecIdxs = []
        for i, comp in enumerate(areas):
            pca = PCA()
            X = np.multiply(comp.coords, np.tile(list(CTimaging.spacing), (len(comp.coords),1))) #TODO: Is that equivalent?!?
            n_samples = X.shape[0]
            X_transformed = pca.fit_transform(X)
            X_centered = X - np.mean(X, axis = 0) # Is this necessary?
            cov_matrix = np.dot(X_centered.T, X_centered) / n_samples # TODO: Does this fit wit the PaCER algorithm?
            latent = pca.explained_variance_ # TODO: Does this fit wit the PaCER algorithm?
            if self.debug: # sanity check
                for latent_test, eigenvector in zip(latent, pca.components_):
                    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

            if len(latent) < 3:
                pass
                #return
            latent = np.sqrt(latent) * 2
            lowerAxesLength = sorted(latent[1:]) # TODO: Does this fit wit the PaCER algorithm?
            if (latent[0] > LAMBDA_1 and latent[0]/np.mean(latent[1:]) > 10
                    and lowerAxesLength[1] / (lowerAxesLength[0] + .001) < 8):
                #elecIdxs.append(componentIdxs[i])
                elecIdxs.append(comp)

        nElecs = len(elecIdxs)
        print('... of which PaCER algorithm guessed {} being electrodes.'.format(nElecs))
        if nElecs == 0:
            if threshold < 3000:
                print('Trying a higher threshold to ensure leads are detected.')
                leadPointcloudStruct, brainMask = self.electrodeEstimation(fileID, CTimaging, brainMask, threshold)
                return
            else:
                print('Even after changing thresholds repeatedly, the algorithms could not detect any leads. Please '
                      'double-check the input carefully ')
                return

   # elecsPointcloudStruct = struct();

    #for i=1:nElecs
    #elecsPointcloudStruct(i).pixelIdxs = ccProps(elecIdxs(i)).PixelIdxList;
    #pixelList = ccProps(elecIdxs(i)).PixelList;
    #pixelList(:, [1 2 3]) = pixelList(:, [2 1 3]); # Swap i, j to X, Y FIXME check this!
    #elecsPointcloudStruct(i).pixelValues = ccProps(elecIdxs(i)).PixelValues

    #elecsPointcloudStruct(i).pointCloudMm = (pixelList - 1) * abs(niiCT.transformationMatrix(1:3, 1: 3)); % minus 1 is done in the get funtions but manually here
    #elecsPointcloudStruct(i).pointCloudWorld = fx_transpose(niiCT.getNiftiWorldCoordinatesFromMatlabIdx(fx_transpose(pixelList))); #
    #bsxfun( @ plus, (pixelList - 1) * niiCT.transformationMatrix(1: 3, 1: 3), fx_transnpose(niiCT.transformationMatrix(1: 3, 4)));

    # elecsPointcloudStruct(i).surroundingPoints = setdiff(bbPointCloud, pixelList, 'rows') * abs(
    #niiCT.transformationMatrix(1:3, 1: 3));

    #elecMask = false(size(maskedImg)); # FIXME check this swaps!!
    #elecMask(elecsPointcloudStruct(i).pixelIdxs) = true; # TODO: make sure we don't have to Swap i,j to X,Y here!
    #elecsPointcloudStruct(i).binaryMaskImage = elecMask;
    #end


    def connected_objects(self, data_array, connectivity):
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

# CT-standart range check (e.g. -1024, 4096)

# identifaction of total metal components in imaging (Metal_Threshold)

# determine brain mask -> available or not ? If yes access to parameters

# detection of metal artifacts inside images, start with maskedimage as complete image without process

# removal of the skull -> accentuate of brainmMask just to show the brain without bony structure

# merge the largest connected components of metal inside the brain

# specificate  number of connected components and confirmation of connection

# access to pixel list and sorting them (Pixel values? Size?)

# [... step will follow]

# guessing the number and idxs of electrodes in image

# determine the mean of voxelnumber

# definition of axis length

# if nElecs=0 error display

# create output structure and try associate electrodes from xml defitions

# with pixelList presentation of Electrode location
