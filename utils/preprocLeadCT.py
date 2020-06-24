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
import numpy as np
import shutil
from scipy import ndimage
from itertools import groupby
from operator import itemgetter
from dependencies import ROOTDIR

#constants
METAL_THRESHOLD = 800
LAMBDA_1 = 25;

class LeadWorks:
    """this class contains all functions needed to detect leads and estimate their rotation angle. All scripts are based
    on the PACER algorithm (https://github.com/adhusch/PaCER)"""

    def __init__(self):
        self.cfg = HF.LittleHelpers.load_config(ROOTDIR)
        self.verbose = 0

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

        fileID =[]
        [fileID.extend(x) for x in allfiles
         if re.search(r'\w.({}[0-9]).'.format(regex_complete), x[0], re.IGNORECASE)
         and x[0].endswith('.nii') and 'CT' in x[0]]

        # here some code is needed which ensures that only one and only one file was found
        originalCT = ants.image_read(fileID[0])
        # TODO: double-check with PACER algorithm if same CT is loaded
        if max(originalCT.spacing) > 1:
            print('Slice thickness > 1 mm! Independent contact detection most likely impossible. '
                  'Using contactAreaCenter based method.')
            flag = 'contactAreaCenter'
        elif max(originalCT.spacing) > .7:
            print('Slice thickness > 0.7 mm! reliable contact detection not guaranteed. However, for certain '
                  'electrode types with large contacts, it might work.')

        min_orig, max_orig = originalCT.min(), originalCT.max()
        rescaleValues = ants.contrib.RescaleIntensity(-1024, 1024)  # to avoid values <0 causing problems w/ log data; maybe wrong and should rather be in if else condition as in Pacer
        rescaleCT = rescaleValues.transform(originalCT)

        # Start with electrode estimation
        cc = self.electrodeEstimation(fileID, rescaleCT, brainMask='', flag='')

    def electrodeEstimation(self, fileID, CTimaging, brainMask, flag, threshold=''):
        """ estimates the electrode mask according to
        https://github.com/adhusch/PaCER/blob/master/src/Functions/extractElectrodePointclouds.m"""

        if not threshold:
            threshold = METAL_THRESHOLD
            print('Routine estimation of number and location of DBS-leads \w standard threshold ({})'.format(threshold))
        else:
            print('Estimation of number and location of DBS-leads \w modified threshold ({})'.format(threshold))

        if CTimaging.dimension !=3:
            print('Something went wrong during CT-preprocessing. There are not three dimensions, please double-check')
            return

        brainMask = np.zeros(CTimaging.shape) # TODO: create a true brainMask
        brainMask[120:150, 100:300, 75:190] =1
        # TODO: double-check size/dimensions of brainMask with PACER
        print('Thresholding {}: {} for artifacts/content with HU > {}'.format(fileID[1], os.path.split(fileID[0])[1],
                                                                              threshold))
        sphere_test = np.array(self.sphere(math.ceil(3/max(CTimaging.spacing))))
        brainMask = ndimage.binary_dilation(brainMask, structure=sphere_test).astype(brainMask.dtype) # TODO: change this to one line command
        brainMask = np.array(brainMask, dtype=bool)

        dataImage_array = CTimaging.numpy()
        dataImage_array[~brainMask] = np.nan

        threshold_image = [dataImage_array > threshold] # TODO: Why is that necessary?

        # largest connected components of metal inside of brain represents the electrodes
        cc = self.connected_objects(threshold_image, connectivity=26)
        print('{} potential metal components were detected within the brain'. format('26'))
        # disp([num2str(cc.NumObjects) ' potential metal components detected within brain.']);
        # replacement for regionprops?!?
        #ccProps = regionprops(cc, niiCT.img, 'Area', 'PixelIdxList', 'PixelList', 'PixelValues', 'BoundingBox');
        #[areas, idxs] = sort([ccProps.Area], 'descend'); % sort by size

        CTimaging = CTimaging.new_image_like(dataImage_array)


        return CTimaging

    def electrode_count(self, fileID, CTimaging, brainMask, threshold, areas, indices):
        """estimate the number of electrodes """

        # In a first step, the detected components are sorted according to size and only those components are selected
        # that have a specific content within the brain

        elecIdxs = []
        minVoxelNumber = (1.2 * 1.27 / 2) ** 2 * math.pi * 40 / np.prod(CTimaging.spacing) # TODO: is niiCT.voxsize really the same thing as CTimaging.spacing? # according to PACER script, 40 mm in the brain and  20 % partial voluming
        maxVoxelNumber = (3 * 1.27 / 2) ** 2 * math.pi * 80 / np.prod(CTimaging.spacing)   # TODO: Is that value identical to PACER?   # assuming 80 mm in brain and 300 % partial voluming

        if self.debug:
            # figure, scatterMatrix3(ccProps(1).PixelList)
            pass

        largeComponents = areas(areas >= minVoxelNumber & areas <= maxVoxelNumber) # Voxels
        componentIdxs = indices(areas >= minVoxelNumber & areas <= maxVoxelNumber)

        for i, comp in enumerate(largeComponents):
            #[~, ~, latent] = pca(ccProps(componentIdxs(i)).PixelList. * repmat(fx_transpose(niiCT.voxsize) , length(ccProps(componentIdxs(i)).PixelList) ,1))
            if len(latent) < 3:
                return
            latent = np.sqrt(latent) * 2
            lowerAxesLength = latent[1:2].sorted() # TODO: Does this fit wit the PaCER algorithm?
            if (latent[0] > LAMBDA_1 &
                    latent[0]/np.mean(latent[1:2]) > 10 &
                    lowerAxesLength[1] / (lowerAxesLength[0] + .001) < 8):
                elecIdxs.append(componentIdxs[i])

        nElecs = len(elecIdxs)
        print('... of which PaCER algorithm guessed {} being electrodes.'.format(nElecs))
        if nElecs == 0:
            if threshold < 3000:
                print('Trying a higher threshold to ensure leads are detected.')
                leadPointcloudStruct, brainMask = self.electrodeEstimation(fileID, CTimaging, brainMask, flag, threshold)
                return
            else:
                print('Even after changing thresholds repeatedly, the algorithms could not detect any leads. Please '
                      'double-check the input carefully ')
                return

    elecsPointcloudStruct = struct();

    for i=1:nElecs
    elecsPointcloudStruct(i).pixelIdxs = ccProps(elecIdxs(i)).PixelIdxList;
    pixelList = ccProps(elecIdxs(i)).PixelList;
    pixelList(:, [1 2 3]) = pixelList(:, [2 1 3]); # Swap i, j to X, Y FIXME check this!
    elecsPointcloudStruct(i).pixelValues = ccProps(elecIdxs(i)).PixelValues

    elecsPointcloudStruct(i).pointCloudMm = (pixelList - 1) * abs(niiCT.transformationMatrix(1:3, 1: 3)); % minus 1 is done in the get funtions but manually here
    elecsPointcloudStruct(i).pointCloudWorld = fx_transpose(niiCT.getNiftiWorldCoordinatesFromMatlabIdx(fx_transpose(pixelList))); #
    bsxfun( @ plus, (pixelList - 1) * niiCT.transformationMatrix(1: 3, 1: 3), fx_transnpose(niiCT.transformationMatrix(1: 3, 4)));

    # elecsPointcloudStruct(i).surroundingPoints = setdiff(bbPointCloud, pixelList, 'rows') * abs(
    niiCT.transformationMatrix(1:3, 1: 3));

    elecMask = false(size(maskedImg)); # FIXME check this swaps!!
    elecMask(elecsPointcloudStruct(i).pixelIdxs) = true; # TODO: make sure we don't have to Swap i,j to X,Y here!
    elecsPointcloudStruct(i).binaryMaskImage = elecMask;
    end


    def connected_objects(self, data_array, connectivity):
        """ function creating a list of objects that are connected and satisfy certain conditions. This aims at
        replacing Mathworks bwconncomp.m function https://www.mathworks.com/help/images/ref/bwconncomp.html"""
        import cc3d

        labels_out = cc3d.connected_components(np.array(data_array), connectivity=connectivity)
        return labels_out

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
