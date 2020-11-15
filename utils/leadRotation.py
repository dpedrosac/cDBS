#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
import os
import warnings

import ants
import numpy as np
import scipy

from dependencies import ROOTDIR
from utils.HelperFunctions import Configuration, LeadWorks


class PrepareData:
    """Plots results from the electrode reconstruction/model creation in order to validate the results"""

    def __init__(self):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug = False

    def getData(self, subj, inputfolder, side='right'):
        """loads all necessary data in order to process it later"""

        filename_leadmodel = os.path.join(os.path.join(inputfolder, subj, 'elecModels_' + subj + '.pkl'))
        if not os.path.isfile(filename_leadmodel):
            print("\n\t...Lead data not found, please make sure it is available")
            return

        lead_data_raw, iP, sS = LeadWorks.load_leadModel(inputfolder, filename_leadmodel)
        lead_data, sS, iP, sides = LeadWorks.estimate_hemisphere(lead_data_raw, sS, iP)  # determines which side is used in lead_data
        single_lead = lead_data[side]

        if lead_data[side]['model'] not in ('Boston Vercise Directional', 'St Jude 6172', 'St Jude 6173'):
            print("\n\t...Rotation not required given 'non-directional' leads; Setting both rotation values to '0'")
            for s in sides:
                lead_data[s]['rotation'] = 0
            LeadWorks.save_leadModel(lead_data, iP, sS)
            rotation_results = dict()
            return rotation_results
        else:
            # lead_model, default_positions, default_coordinates = LeadWorks.get_default_lead(lead_data_raw[0])  # all in [mm] #TODO necessary at all???
            CTimaging = ants.image_read(filename=os.path.join(*lead_data['left']['filenameCTimaging']))

        slices, boxes, dir_level = GetImaging.extractAxialSlices(single_lead, pixVal=1, extractradius=30, offset=0)
        angles, intensities_marker_angle, vector_marker = GetImaging.orient_intensityprofile(slices['marker'],
                                                                                             pixdim=CTimaging.spacing,
                                                                                             radius=8)
        levels = dict([(k, r) for k, r in slices.items() if k.startswith('level')])
        angles, intensities_level_angle, vectors_levels = [{k: [] for k in levels} for _ in range(3)]
        for lv in levels:
            _, intensities_level_angle[lv], vectors_levels[lv] = \
                GetImaging.orient_intensityprofile(slices[lv], pixdim=CTimaging.spacing, radius=16)

        [peak, markerfft] = GetImaging.orient_intensityPeaks(intensities_marker_angle)
        default = 'anterior'
        if default == 'anterior':
            finalpeak = peak[1] if peak[0] > 90 or peak[0] < 270 else peak[0]
        elif default == 'posterior':
            finalpeak = peak[0] if peak[0] > 90 or peak[0] < 270 else peak[1]

        # Calculate yaw and pitch angles for correction at the end
        yaw = math.asin(single_lead['normtraj_vector'][0])
        pitch = math.asin(single_lead['normtraj_vector'][1] / math.cos(yaw))

        warningtext = "Warning: {} > 40 deg - Determining orientation might be inaccurate!"
        if np.rad2deg(abs(pitch)) > 40:
            warnings.warn(warningtext.format('Pitch'))
        if np.rad2deg(abs(yaw)) > 40:
            warnings.warn(warningtext.format('Yaw'))

        peakangle = np.deg2rad(finalpeak)  # TODO: not sure if that is always right, double check!!
        peakangle_corrected = (math.sin(peakangle) * math.cos(pitch)) / \
                              ((math.cos(peakangle) * math.cos(yaw)) -
                               (math.sin(peakangle) * math.sin(yaw) * math.sin(pitch)))  # Sitz et al. 2017
        peakangle_corrected = np.arctan(peakangle_corrected)

        if peakangle < math.pi and peakangle_corrected < 0 and (peakangle - peakangle_corrected) > math.pi/2:
            peakangle_corrected = peakangle_corrected + math.pi

        if peakangle > math.pi and peakangle_corrected > 0 and (peakangle - peakangle_corrected) > math.pi/2:
            peakangle_corrected = peakangle_corrected - math.pi

        roll = peakangle_corrected

        sum_intensity, roll, dir_valleys = GetImaging.determine_segmented_orientation(intensities_level_angle, roll, yaw, pitch, dir_level)

        return slices, intensities_marker_angle, intensities_level_angle, sum_intensity, roll, dir_valleys, markerfft
        # TODO at this point a list of all necessary returns is required to create the data which is needed for plotting

        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # fig.add_subplot(1,3,1)
        # plt.imshow(slices['marker'])
        # fig.add_subplot(1,3,2)
        # plt.imshow(slices['level1'])
        # fig.add_subplot(1,3,3)
        # plt.imshow(slices['level2'])
        #
        # plt.figure()
        # fig.add_subplot(1,3,1)
        # plt.plot(intensities_marker_angle)
        # fig.add_subplot(1,3,2)
        # plt.plot(intensities_level_angle['level1'])
        # fig.add_subplot(1,3,3)
        # plt.plot(intensities_level_angle['level2'])


class GetImaging:
    """Class to estimate necessary information to plot the results later"""

    def __init__(self, _debug=False):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug = _debug

    @staticmethod
    def extractAxialSlices(single_lead, pixVal, extractradius=30, offset=0, levels=('level1', 'level2')):

        lead_settings = {'Boston Vercise Directional': {'markerposition': 12, 'leadspacing': 2},
                         'St Jude 6172': {'markerposition': 9, 'leadspacing': 2},
                         'St Jude 6173': {'markerposition': 12, 'leadspacing': 3}}

        all_marker = dict([(k, r) for k, r in single_lead.items() if k.startswith('marker')])
        unit_vector = single_lead['normtraj_vector']  # see preprocLeadCT for further details
        marker_mm = np.round(all_marker['markers_head'] + [0, 0, offset] +
                             np.multiply(unit_vector, lead_settings[single_lead['model']]['markerposition']/pixVal))

        intensity, box, dir_level = [dict() for _ in range(3)]
        intensity['marker'], box['marker'], _ = GetImaging.get_axialplanes(marker_mm, single_lead,
                                                                           window_size=extractradius)
        for idx, l in enumerate(levels, start=1):
            dir_level[l] = np.round(all_marker['markers_head'] + [0, 0, offset] +
                                    np.multiply(unit_vector, idx * lead_settings[single_lead['model']]['leadspacing'] / pixVal))
            intensity[l], box[l], _ = GetImaging.get_axialplanes(dir_level[l], single_lead, window_size=extractradius)

        return intensity, box, dir_level

    @staticmethod
    def get_axialplanes(marker_coordinates, single_lead, window_size=15, resolution=.5, transformation_matrix=np.eye(4)):
        """returns a plane at a specific window with a certain direction"""

        bounding_box_coords = []
        for k in range(2):
            bounding_box_coords.append(np.arange(start=marker_coordinates[k]-window_size,
                                                 stop=marker_coordinates[k]+window_size, step=resolution))
        bounding_box_coords.append(np.repeat(marker_coordinates[-1], len(bounding_box_coords[1])))
        bounding_box = np.array(bounding_box_coords)

        meshX, meshY = np.meshgrid(bounding_box[0, :], bounding_box[1, :])
        meshZ = np.repeat(bounding_box[-1,0], len(meshX.flatten()))
        fitvolume_orig = np.array([meshX.flatten(), meshY.flatten(), meshZ.flatten(), np.ones(meshX.flatten().shape)])
        fitvolume = np.linalg.solve(transformation_matrix, fitvolume_orig)
        resampled_points = GetImaging.interpolate_CTintensities(single_lead, fitvolume)
        imat = np.reshape(resampled_points, (meshX.shape[0], -1), order='F')

        return imat, bounding_box, fitvolume

    @staticmethod
    def interpolate_CTintensities(single_lead, fitvolume):
        import SimpleITK as sitk

        img = sitk.ReadImage(os.path.join(*single_lead['filenameCTimaging']))
        physical_points = list(map(tuple, fitvolume[0:3, :].T))
        # physical_points = physical_points[0:5]
        num_samples = len(physical_points)
        physical_points = [img.TransformContinuousIndexToPhysicalPoint(pnt) for pnt in physical_points]

        # interp_grid_img = sitk.Image((len(physical_points) *([1] * (img.GetDimension() - 1))), sitk.sitkUInt8)
        interp_grid_img = sitk.Image([num_samples] + [1] * (img.GetDimension() - 1), sitk.sitkUInt8)
        displacement_img = sitk.Image([num_samples] + [1] * (img.GetDimension() - 1), sitk.sitkVectorFloat64,
                                      img.GetDimension())

        for i, pnt in enumerate(physical_points):
            displacement_img[[i] + [0] * (img.GetDimension() - 1)] = np.array(pnt) - np.array(
                interp_grid_img.TransformIndexToPhysicalPoint([i] + [0] * (img.GetDimension() - 1)))

        interpolator_enum = sitk.sitkLinear
        default_output_pixel_value = 0.0
        output_pixel_type = sitk.sitkFloat32 if img.GetNumberOfComponentsPerPixel() == 1 else sitk.sitkVectorFloat32
        resampled_temp = sitk.Resample(img, interp_grid_img, sitk.DisplacementFieldTransform(displacement_img),
                                       interpolator_enum, default_output_pixel_value, output_pixel_type)

        resampled_points = [resampled_temp[x, 0, 0] for x in range(resampled_temp.GetWidth())]
        debug = False
        if debug:
            for i in range(resampled_temp.GetWidth()):
                print(str(img.TransformPhysicalPointToContinuousIndex(physical_points[i])) + ': ' + str(
                    resampled_temp[[i] + [0] * (img.GetDimension() - 1)]) + '\n')

        return np.array(resampled_points)

    @staticmethod
    def orient_intensityprofile(artefact_slice, pixdim, radius=16, center=''):
        """estimates the intensities at the different angles aroud the artefact; all functions are derived from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensityprofile.m"""

        center = np.array([len(artefact_slice)/2, len(artefact_slice)/2]) if not center else center
        vector = np.multiply([0, 1], radius / pixdim[-1])
        vector_updated, angle, intensity = [[] for _ in range(3)]
        for k in range(1, 361):
            theta = (2 * math.pi / 360) * (k - 1)
            rotation_matrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            vector_updated.append(vector @ rotation_matrix + center)
            angle.append(theta)
            int_temp = GetImaging.interpolate_image(artefact_slice, vector_updated[-1])
            intensity.append(int_temp)

        return angle, intensity, vector_updated

    @staticmethod
    def interpolate_image(artefact_slice, yx, zpad=True, RGB=1):
        """BiLinear interpolation using 4 pixels around the target location with ceil convention, adapted from 
        https://github.com/netstim/leaddbs/blob/master/ea_orient_interpimage.m"""

        yx0 = np.floor(yx)
        wt = yx - yx0
        wtConj = 1 - wt

        interTop = wtConj[1] * GetImaging.pixLookup(artefact_slice, yx0[0], yx0[1], RGB=1) + wt[1] * \
                   GetImaging.pixLookup(artefact_slice, yx0[0], yx[1], RGB=1)
        interBtm = wtConj[1] * GetImaging.pixLookup(artefact_slice, yx[0], yx0[1], RGB=1) + wt[1] * \
                   GetImaging.pixLookup(artefact_slice, yx[0], yx[1], RGB=1)
        interVal = wtConj[0] * interTop + wt[0] * interBtm

        return interVal

    @staticmethod
    def pixLookup(artefact_slice, y, x, zpad=True, RGB=1):
        """function which looks up a pixel value from a given image
        @params:
            artefact_slice  - Required  : intensities at a give axial slice
            y/x             - Required  : coordinates in x- and y-axis
        """

        pixVal = np.zeros(shape=[1, 1, RGB])
        if RGB == 3:
            ROW, COL, _ = artefact_slice.shape
        else:
            ROW, COL = artefact_slice.shape

        if x <= 0 or x > COL or y <= 0 or y > ROW:
            if zpad:
                pixVal[:] = 0
            else:
                y0, x0 = y, x
                y0[y0 < 1], x0[x0 < 1] = 1, 1
                y0[y0 > ROW] = ROW
                x0[x0 > COL] = COL
                pixVal = artefact_slice[int(np.ceil(y0)), int(np.ceil(x0))]
        else:
            pixVal = artefact_slice[int(np.ceil(y)), int(np.ceil(x))]  # third dimension removed as it is not required

        return pixVal

    @staticmethod
    def orient_intensityPeaks(intensity, noPeaks=2):
        """estimates peaks for the intensities obtaines before; function derived from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensitypeaksFFT.m"""

        fft_intensities = scipy.fft(intensity)
        fft_part = fft_intensities[noPeaks+1]  # TODO is it really necessary to add 1 in consideration of different indexing between Matlab and Python?
        phase = math.asin(np.real(fft_part) / abs(fft_part))

        if np.imag(fft_part) > 0:
            phase = -math.pi - phase if np.real(fft_part) > 0 else math.pi - phase

        amplitude = (np.max(intensity) + np.abs(np.min(intensity))) / 2
        level = np.max(intensity) - amplitude
        sprofil, peak, sum_intensity = [[] for _ in range(3)]
        for k in range(1, 361):
            sprofil.append(amplitude * math.sin(np.deg2rad(noPeaks * k) - phase) + level)

        [peak.append(x*360/noPeaks) for x in range(noPeaks)]

        no_iter = 360/noPeaks
        for k in range(int(no_iter)):
            sum_intensity.append(np.sum([sprofil[int(i)] for i in peak]))
            peak = np.add(peak, 1)

        maxpeak = np.argmax(sum_intensity)

        for k in range(noPeaks):
            peak[k] = maxpeak + k*360/noPeaks

        return peak, sprofil

    @staticmethod
    def determine_segmented_orientation(intensities, roll, yaw, pitch, dir_level):
        """ determine angles of the 6-valley artifact ('dark star') artifact around directional markers; for details cf.
        https://github.com/netstim/leaddbs/blob/master/ea_orient_main.m (lines 474f.)"""

        sum_intensities, roll_values, dir_valleys = [{k: [] for k in intensities.keys()} for _ in range(3)]
        rollangles = []
        for level, intensity in intensities.items():
            count = 0
            shift = []

            for k in range(-30, 31):
                shift.append(k)
                rolltemp = roll + np.deg2rad(k)
                dir_angles = GetImaging.orient_artifact_at_level(rolltemp, pitch, yaw, dir_level[level])
                temp = GetImaging.peaks_dir_marker(intensity, dir_angles)
                sum_intensities[level].append(temp)
                if level == list(intensities.keys())[0]:
                    rollangles.append(rolltemp)
                count += 1

        dir_angles = {k: [] for k in intensities.keys()}
        for level in intensities.keys():
            temp = np.argmin(sum_intensities[level])
            roll_values[level] = rollangles[temp]
            dir_angles[level] = GetImaging.orient_artifact_at_level(roll_values[level], pitch, yaw, dir_level[level])
            dir_valleys[level] = np.round(np.rad2deg(dir_angles[level]) + 1)
            dir_valleys[level][dir_valleys[level] > 360] = dir_valleys[level][dir_valleys[level] > 360] - 360

        return sum_intensities, roll_values, dir_valleys

    @staticmethod
    def peaks_dir_marker(intensity, angles):
        """function aiming at detecting the 'intensity peaks'; search is restricted to 360°/noPeaks. adapted from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensitypeaksdirmarker.m """

        intensities = np.array(intensity)
        peak = np.round(np.rad2deg(angles))
        # peak[peak<1] = peak[peak<1] + 360
        peak[peak > 359] = peak[peak > 359] - 359
        idx_peaks = [int(x) for x in peak]

        return np.sum(intensities[idx_peaks])

    @staticmethod
    def orient_artifact_at_level(roll, pitch, yaw, dir_level, degrees=(60, 180, 300)):
        """function aiming at detecting the 'intensity peaks'; search is restricted to 360°/noPeaks. adapted from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensitypeaksdirmarker.m """

        dir_level = dir_level[0:3]
        ventral_default = [0, 0.65, -0.75]
        dorsal_default = [0, 0.65, 0.75]

        ventral, dorsal, vec, direction = [{k: [] for k in degrees} for _ in range(4)]
        for idx, deg in enumerate(degrees):
            M, _, _, _ = GetImaging.orient_rollpitchyaw(roll - (deg / 60 * (2 * math.pi) / 6), pitch, yaw)
            ventral[deg] = M @ ventral_default
            dorsal[deg] = M @ dorsal_default

        # calculate intersecting points between vec60 / 180 / 300 and the z - plane through the dir - level artifact
        for idx, deg in enumerate(degrees):
            vec[deg] = np.divide(np.subtract(dorsal[deg], ventral[deg]),
                                 np.linalg.norm(np.subtract(dorsal[deg], ventral[deg])))  # unitvector venXdeg 2 dorXdeg
            dir_ventral_temp = np.add(dir_level, ventral[deg])  # ventral point @ X° from directional level
            # dir_dorsal_temp = np.add(dir_level, dorsal[deg])  # dorsal point @ X° from directional level -> plotting
            dir_xdegrees = (dir_level[-1] - dir_ventral_temp[-1]) / vec[deg][-1]
            direction[deg] = dir_ventral_temp + np.multiply(dir_xdegrees, vec[deg])

        # create vectors corresponding to the dark lines of the artefacts
        allcombs = list(itertools.combinations(degrees, 2))
        dir_vec = []
        for fst, scnd in allcombs:
            dir_vec.append((direction[fst] - direction[scnd]) / np.linalg.norm(direction[fst] - direction[scnd]))

        # calculate the angles of the dark lines with respect to the y-axis
        dir_angles = []
        for idx, deg in enumerate(degrees):
            dir_angles.append(np.arctan2(np.linalg.norm(np.cross(dir_vec[idx], [0, 1, 0])),
                                         np.dot(dir_vec[idx], [0, 1, 0])))
            dir_angles[idx] = -dir_angles[idx] if dir_vec[idx][0] < 0 else dir_angles[idx]

        dir_angles.extend(np.add(dir_angles, math.pi))
        dir_angles = np.asarray(dir_angles, dtype=np.float32)
        dir_angles[dir_angles > 2*math.pi] = dir_angles[dir_angles > 2 * math.pi] - 2 * math.pi
        dir_angles[dir_angles < 0] = dir_angles[dir_angles < 0] + 2 * math.pi
        dir_angles = np.sort(2 * math.pi-dir_angles)

        return dir_angles

    @staticmethod
    def orient_rollpitchyaw(roll, pitch, yaw):
        """function returning a 3x3 rotation matrix for roll, pitch and yaw rotation [all in radians] for further
        details cf. https://github.com/netstim/leaddbs/blob/master/ea_orient_rollpitchyaw.m """

        a, b, c = pitch, yaw, roll
        Mx = np.array([1, 0, 0, 0, math.cos(a), math.sin(a), 0, -math.sin(a), math.cos(a)]).reshape(3, 3)
        My = np.array([math.cos(b), 0, math.sin(b), 0, 1, 0, -math.sin(b), 0, math.cos(b)]).reshape(3, 3)
        Mz = np.array([math.cos(c), -math.sin(c), 0, math.sin(c), math.cos(c), 0, 0, 0, 1]).reshape(3, 3)

        M = (Mx @ My) @ Mz
        return M, Mz, My, Mx
