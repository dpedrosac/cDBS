#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
import os
import re
import warnings

import ants
import numpy as np
import scipy

from dependencies import ROOTDIR, lead_settings
from utils.HelperFunctions import Configuration, LeadProperties

cfg = Configuration.load_config(ROOTDIR)


def function_wrapper(subj, side, lead_data=''):
    """function calling all necessary steps; i.e. this function when called when initiating script is used to gather
    all data, whereas for special functionality other modules can be called/used. The steps are as follows:
        a) load_LeadData, b) get_CTimaging, c) axialSlices (cf. lead_details) per level, d) get_intensities,
        e) rotation_estimate """

    input_folder = cfg['folders']['nifti'] # default NIFTI-folder
    if not lead_data:
        lead_data, _, _ = General.load_leadData(subj, input_folder=input_folder, side=side)

    if lead_data['model'] not in ('Boston Vercise Directional', 'St Jude 6172', 'St Jude 6173'):
        print("\n\t...Rotation not required given 'non-directional' leads")
        rotation = {'angle': -math.pi / 10}
        return rotation # rotation not necessary for non-directional leads

    CTimaging_orig, CTimaging_trans = General.get_CTimaging(lead_data)
    rotation, UnitVector = General.lead_details(subj, input_folder, side=side, x_offset=[0]*3,
                                                level_names=['marker', 'level1', 'level2'],
                                                rotation=Configuration.rotation_dict_mod(),
                                                single_lead=lead_data, CTimaging_orig=CTimaging_orig,
                                                CTimaging_trans=CTimaging_trans)

    angles, intensities_angle, vector, peak, markerfft, valleys, finalpeak = \
        [{k: [] for k in rotation['slices'].keys()} for _ in range(7)]  # Pre-allocate data
    for lvl in rotation['slices'].keys():
        angles[lvl], intensities_angle[lvl], vector[lvl], peak[lvl], markerfft[lvl], valleys[lvl], finalpeak[lvl] = \
            General.get_intensities(lvl, rotation['slices'][lvl], CTimaging_trans)

    rotation['vector'] = vector
    rotation['markerfft'] = markerfft['marker']
    rotContact_coords = dict([(k, r) for k, r in rotation['coordinates'].items() if k.startswith('level')])
    rotation = General.rotation_estimate(UnitVector, angles['marker'], finalpeak['marker'], intensities_angle,
                                         rotContact_coords, valleys['marker'], CTimaging_orig, rotation)
    return rotation


class General:
    """ General functions aiming at getting already processed information and estimating rotation"""

    def __init__(self, _debug=False):
        self.debug = _debug

    @staticmethod
    def load_leadData(subj, input_folder, side='right'):
        """loads lead data if already available, for details cf. ~/utils/preprocLeadCT.py"""

        filename_leadmodel = os.path.join(os.path.join(input_folder, subj, 'elecModels_' + subj + '.pkl'))
        if not os.path.isfile(filename_leadmodel):
            print("\n\t...Lead data not found, please make sure it is available")
            return

        lead_data_raw, iP, sS = LeadProperties.load_leadModel(input_folder, filename_leadmodel)
        lead_data, sS, iP, sides = LeadProperties.estimate_hemisphere(lead_data_raw, sS, iP)  # determines side, TODO: after tidying up the code, swap outputs
        return lead_data[side], iP[side], sS[side]

    @staticmethod
    def get_CTimaging(lead_data):
        """load and retirn CTimaging as ANTs object of both: original space and transformed spaced"""

        CTimaging_trans = ants.image_read(filename=os.path.join(*lead_data['filenameCTimaging']))
        CTimaging_orig = ants.image_read('/'.join(re.split(r".reg_run[0-9].",
                                                               os.path.join(*lead_data['filenameCTimaging']))))
        return CTimaging_orig, CTimaging_trans

    @staticmethod
    def lead_details(subj, input_folder, side='right', x_offset=[0]*3, level_names=['marker', 'level1', 'level2'],
                     rotation='', single_lead='', CTimaging_orig='', CTimaging_trans=''):
        """wrapper to retrieve default data for coordinates, slices, fitvolumes and boxes
        which are sorted in a sequential fashion; enabling changes as desired at later stages"""

        # Pre-allocate space and get 'empty' rotation dictionary if no input was provided
        if not rotation:
            rotation = Configuration.rotation_dict_mod()

        if not single_lead:
            single_lead, _, _ = General.load_leadData(subj, input_folder=input_folder, side=side)

        if not CTimaging_orig or not CTimaging_trans:
            CTimaging_orig, CTimaging_trans = General.get_CTimaging(single_lead)

        slices, boxes, fitvolumes, marker_coord, directional_coord, UnitVector = [dict() for _ in range(6)]
        for idx, k in enumerate(level_names):
            slices[k], boxes[k], fitvolumes[k], directional_coord[k], UnitVector = \
                Specific.AxialSlices_levelwise(single_lead, CTimaging_trans, CTimaging_orig, x_offset=x_offset[idx],
                                               extractradius=30, part2lookfor=k, level_dist=int(idx))

        # according to original script, slices [artifact_marker, artifact_dirX] MUST be flipped. Rationale for this
        # is not quite clear; cf. https://github.com/netstim/leaddbs/blob/master/ea_orient_main.m lines 138ff.
        for k, v in slices.items():
            slices[k] = np.fliplr(v)

        # Update and return rotation dictionary for further processing
        for level in level_names:
            rotation['coordinates'][level] = directional_coord[level]
            rotation['plot_box'][level] = boxes[level]
            rotation['slices'][level] = slices[level]
            rotation['fitvolumes'][level] = fitvolumes[level]

        return rotation, UnitVector

    @staticmethod
    def get_intensities(level, slices, CTimaging_trans, radius=8, default_direction='anterior'):
        """generic function intended at providing the intensities for either the marker or the distinct levels and
        corresponding further data"""

        peak, markerfft, valley_marker, finalpeak = [[] for _ in range(4)] # Pre-allocate space
        radius = 16 if level != 'marker' else radius  # if level is selected for directional leads, radius is increased

        angles, intensities_angle, vector = \
            Specific.orient_intensityprofile(slices, pixdim=CTimaging_trans.spacing, radius=radius)

        if level == 'marker':
            [peak, markerfft] = Specific.orient_intensityPeaks(intensities_angle)
            [valley_marker, _] = Specific.orient_intensityPeaks([element * -1 for element in intensities_angle])

            if default_direction == 'anterior':
                finalpeak = peak[1] if 90 < peak[0] < 270 else peak[0]
            elif default_direction == 'posterior':
                finalpeak = peak[0] if 90 < peak[0] < 270 else peak[1]

        return angles, intensities_angle, vector, peak, markerfft, valley_marker, finalpeak

    @staticmethod
    def rotation_estimate(UnitVector, angles, finalpeak, intensities_angle, rotContact_coords,
                          valley_marker, CTimaging_orig, rotation):
        """function which returns rotation angles for lead @ marker and directional lead depths"""

        if False in [bool(intensities_angle.get(x)) for x in intensities_angle.keys()]:
            warnings.warn(message='intensity values are incomplete, hence rotation estimation not possible')
            return rotation

        intensities_level = dict([(k, r) for k, r in intensities_angle.items() if k.startswith('level')])
        intensities_marker = dict([(k, r) for k, r in intensities_angle.items() if k.startswith('marker')])

        yaw = math.asin(UnitVector[0])
        pitch = math.asin(UnitVector[1] / math.cos(yaw))

        warningtext = "Warning: {} > 40 deg - Determining orientation might be inaccurate!"
        if np.rad2deg(abs(pitch)) > 40:
            warnings.warn(warningtext.format('pitch'))
        if np.rad2deg(abs(yaw)) > 40:
            warnings.warn(warningtext.format('yaw'))

        peakangle = angles[int(finalpeak)]  # TODO: not sure if that is always right, double check!!
        peakangle_corrected = (math.sin(peakangle) * math.cos(pitch)) / \
                              ((math.cos(peakangle) * math.cos(yaw)) -
                               (math.sin(peakangle) * math.sin(yaw) * math.sin(pitch)))  # Sitz et al. 2017
        peakangle_corrected = np.arctan(peakangle_corrected)

        if peakangle < math.pi and peakangle_corrected < 0 and (peakangle - peakangle_corrected) > math.pi / 2:
            roll_marker = peakangle_corrected + math.pi
        elif peakangle > math.pi and peakangle_corrected > 0 and (peakangle - peakangle_corrected) > math.pi / 2:
            roll_marker = peakangle_corrected - math.pi
        else:
            roll_marker = peakangle_corrected

        # intensities_levels, directional_coord = # TODO only select those intensities @ levels not where the marker is
        sum_intensity, roll, dir_valleys, roll_angles = \
            Specific.determine_segmented_orientation(intensities_level, roll_marker, yaw, pitch,
                                                     rotContact_coords, CTimaging_orig)

        # Wrap up all results and return a single dictionary for further processing
        intensities = {key: np.array([value, angles]).T for key, value in intensities_level.items()}
        intensities.update({'marker': np.array([intensities_marker['marker'], angles]).T})

        dir_valleys.update({'marker': valley_marker})
        roll.update({'marker': roll_marker})

        rotation = {'intensities': intensities,
                    'sum_intensities': sum_intensity,
                    'valleys': dir_valleys,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll_angles': roll_angles,
                    'angle': np.rad2deg(roll['marker']),
                    'angle_manual_correction': 0}

        return rotation

    @staticmethod
    def manual_angle_correction(rotation):
        """in case angles were corrected manually, this function is run in order to adapt visualisation"""
        print('not yet implemented!!')
        return rotation


class Specific:
    """ Specific functions which have a role for the steps used in General (above)"""

    def __init__(self, _debug=False):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug = _debug

    @staticmethod
    def AxialSlices(single_lead, CTimaging_trans, CTimaging_orig,
                           extractradius=30, offset=[0] * 3, levels=('level1', 'level2')):
        """ extracts intensities of axial slices of the CT. In order to get a consistent artefact
        at the marker, a transformation to the original data is necessary as this corresponds to a steeper angle;
        NOTE THIS FUNCTION WAS REPLACED WITH THE MORE GENERIC axialSlices_levelwise FUNCTION BELOW!"""

        cfg = Configuration.load_config(ROOTDIR)['preprocess']['registration']

        intensity, box, fitvolume, rotContact_coords = [dict() for _ in range(4)]  # pre-allocate for later
        estimated_markers = dict([(k, r) for k, r in single_lead.items() if k.startswith('marker')])
        file_invMatrix = os.path.join(single_lead['filenameCTimaging'][0], 'CT2template_1InvWarpMatrix.mat')

        # Convert the estimated markers (cf. preprocLeadCT.py) to original imaging data w/ transformation matrix
        marker_origCT = dict()  # next part inconsistent to orientation from ea_main_orient (cf. lines 108ff.)!!
        for marker, coordinates in estimated_markers.items():
            coords_temp = [int(round(c)) for c in coordinates]
            transformed = LeadProperties.transform_coordinates(coords_temp, from_imaging=CTimaging_trans,
                                                               to_imaging=CTimaging_orig, file_invMatrix=file_invMatrix)
            marker_origCT[marker] = np.array(transformed['points'])  # use points to account for distances

        UnitVector_origCT = np.divide((marker_origCT['markers_tail'] - marker_origCT['markers_head']),
                                      np.linalg.norm(marker_origCT['markers_tail'] - marker_origCT['markers_head']))
        marker_origCTmm = np.round(marker_origCT['markers_head'] + [0, 0, offset[0]] +
                                   np.multiply(lead_settings[single_lead['model']]['markerposition'],
                                               UnitVector_origCT))
        marker_origCTvx = ants.transform_physical_point_to_index(CTimaging_orig, point=marker_origCTmm)

        filenameCT = '/'.join(re.split(r".{}run[0-9].".format(cfg['prefix']),
                                       os.path.join(*single_lead['filenameCTimaging'])))

        # Extract the intensities from axial slices which result from original imaging and markers in vx-coordinates
        intensity['marker'], box['marker'], fitvolume['marker'] = Specific.get_axialplanes(marker_origCTvx,filenameCT,
                                                                                           UnitVector_origCT,
                                                                                           window_size=extractradius)
        marker_coordinates = marker_origCTvx

        # Loop through directional levels to obtain necessary coordinates and rotation
        for idx, l in enumerate(levels, start=1):
            rotContact_coords[l] = np.round(marker_origCT['markers_head'] + [0, 0, offset[idx]] +
                                            np.multiply(UnitVector_origCT,
                                                        idx * lead_settings[single_lead['model']]['leadspacing']))
            rotContact_coords[l] = ants.transform_physical_point_to_index(CTimaging_orig,
                                                                          point=rotContact_coords[l])
            intensity[l], box[l], fitvolume[l] = Specific.get_axialplanes(rotContact_coords[l], filenameCT,
                                                                          UnitVector_origCT, window_size=extractradius)

        return intensity, box, fitvolume, marker_coordinates, rotContact_coords, UnitVector_origCT

    @staticmethod
    def AxialSlices_levelwise(single_lead, CTimaging_trans, CTimaging_orig, extractradius=30, x_offset=0,
                              part2lookfor='marker', level_dist=1):
        """ extracts intensities of axial CT-slices (levelwise). In order to get consistent artefacts at the marker,
         transformations to original data (CTimaging_orig) are required as this corresponds to steeper angle"""

        # General settings and pre-allocation of space
        cfg = Configuration.load_config(ROOTDIR)['preprocess']['registration']  # specific part of cfg, required later
        file_invMatrix = os.path.join(single_lead['filenameCTimaging'][0], 'CT2template_1InvWarpMatrix.mat')
        filenameCT = '/'.join(re.split(r".{}run[0-9].".format(cfg['prefix']),
                                       os.path.join(*single_lead['filenameCTimaging'])))

        # Convert estimated markers (cf. preprocLeadCT.py) to original imaging data w/ transformation matrix
        estimated_markers = dict([(k, r) for k, r in single_lead.items() if k.startswith('marker')])
        marker_origCT = dict()  # next part inconsistent to orientation from ea_main_orient (cf. lines 108ff.)!!
        for marker, coordinates in estimated_markers.items():
            coords_temp = [int(round(c)) for c in coordinates]
            transformed = LeadProperties.transform_coordinates(coords_temp, from_imaging=CTimaging_trans,
                                                               to_imaging=CTimaging_orig, file_invMatrix=file_invMatrix)
            marker_origCT[marker] = np.array(transformed['points'])  # use points to account for distances

        UnitVector_origCT = np.divide((marker_origCT['markers_tail'] - marker_origCT['markers_head']),
                                      np.linalg.norm(marker_origCT['markers_tail'] - marker_origCT['markers_head']))

        marker_origCTmm = np.round(marker_origCT['markers_head'] + [0, 0, x_offset] +
                                   np.multiply(lead_settings[single_lead['model']]['markerposition'],
                                               UnitVector_origCT))
        marker_origCTvx = ants.transform_physical_point_to_index(CTimaging_orig, point=marker_origCTmm)

        # Extract intensities from axial slices which result from original imaging and markers in vx-coordinates
        if part2lookfor == 'marker':
            intensity, box, fitvolume = \
                Specific.get_axialplanes(marker_origCTvx, filenameCT, UnitVector_origCT, window_size=extractradius)
            coordinates_temp = marker_origCTvx
        else:  # extract from 'levels', i.e. directional contacts
            coordinates_temp = np.round(marker_origCT['markers_head'] + [0, 0, x_offset] +
                                        np.multiply(UnitVector_origCT,
                                                    level_dist * lead_settings[single_lead['model']]['leadspacing']))
            coordinates_temp = ants.transform_physical_point_to_index(CTimaging_orig, point=coordinates_temp)
            intensity, box, fitvolume = Specific.get_axialplanes(coordinates_temp, filenameCT, UnitVector_origCT,
                                                                 window_size=extractradius)

        return intensity, box, fitvolume, coordinates_temp, UnitVector_origCT

    @staticmethod
    def get_axialplanes(lead_point, imaging, unit_vector, window_size=10, res=.5, transformation_matrix=np.eye(4)):
        """returns intensities at perpendicular plane to unit_vector (i.e. trajectory)"""

        perpendicular_vectors = {0: np.random.randn(3), 1: np.random.randn(3)}
        for num, vec in perpendicular_vectors.items():
            perpendicular_vectors[num] -= vec.dot(unit_vector) * unit_vector
            perpendicular_vectors[num] /= np.linalg.norm(perpendicular_vectors[num])

        d = np.dot(unit_vector, lead_point)

        coords_bb = []  # this is the bounding box for the coordinates to interpolate
        for k in range(3):
            coords_bb.append(np.arange(start=lead_point[k] - window_size, stop=lead_point[k] + window_size, step=res))
        bounding_box = np.array(coords_bb)

        meshX, meshY = np.meshgrid(bounding_box[0, :], bounding_box[1, :])
        meshZ = (d - unit_vector[0] * meshX - unit_vector[1] * meshY) / unit_vector[2]

        fitvolume_orig = np.array([meshX.flatten(), meshY.flatten(), meshZ.flatten(), np.ones(meshX.flatten().shape)])
        fitvolume = np.linalg.solve(transformation_matrix, fitvolume_orig)
        resampled_points = LeadProperties.interpolate_CTintensities(fitvolume, imaging, method='polygon')
        imat = np.reshape(resampled_points, (meshX.shape[0], -1), order='F')

        return imat, bounding_box, fitvolume

    @staticmethod
    def orient_intensityprofile(artefact_slice, pixdim, radius=16, center=''):
        """estimates intensities at different angles around the 'artefact'; all functions derived from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensityprofile.m"""

        center = np.array([len(artefact_slice) / 2, len(artefact_slice) / 2]) if not center else center
        vector = np.multiply([0, 1], radius / pixdim[0])  # use pixel dimension in the first axis ('lateral')
        vector_updated, angle, intensity = [[] for _ in range(3)]
        for k in range(1, 361):
            theta = (2 * math.pi / 360) * (k - 1)
            rotation_matrix = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
            vector_updated.append(vector @ rotation_matrix + center)
            int_temp = Specific.interpolate_image(artefact_slice, vector_updated[-1])
            intensity.append(int_temp)
            angle.append(theta)

        return angle, intensity, vector_updated

    @staticmethod
    def interpolate_image(artefact_slice, yx):
        """Bi-Linear interpolation using 4 pixels around the target location with ceil convention, adapted from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_interpimage.m"""

        yx0 = np.floor(yx)
        wt = yx - yx0
        wtConj = 1 - wt
        interTop = wtConj[1] * Specific.pixLookup(artefact_slice, yx0[0], yx0[1], RGB=1) + wt[1] * \
                   Specific.pixLookup(artefact_slice, yx0[0], yx[1], RGB=1)
        interBottom = wtConj[1] * Specific.pixLookup(artefact_slice, yx[0], yx0[1], RGB=1) + wt[1] * \
                      Specific.pixLookup(artefact_slice, yx[0], yx[1], RGB=1)

        return wtConj[0] * interTop + wt[0] * interBottom  # termed interVal in ea_orient_interpimage.m

    @staticmethod
    def pixLookup(artefact_slice, y, x, zpad=True, RGB=1):
        """function which looks up pixel value from given image, according to:
        https://github.com/netstim/leaddbs/blob/master/ea_orient_interpimage.m

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
        """estimates peaks for the intensities obtained before cf. Specific.orient_intensityprofile(); function adapted
        from https://github.com/netstim/leaddbs/blob/master/ea_orient_intensitypeaksFFT.m"""

        fft_intensities = scipy.fft(intensity)
        fft_part = fft_intensities[noPeaks]  # Reason for indexing not quite clear here. results more accurate when not
        phase = -math.asin(np.real(fft_part) / abs(fft_part))

        if np.imag(fft_part) > 0:
            phase = -math.pi - phase if np.real(fft_part) > 0 else math.pi - phase

        amplitude = (np.max(intensity) + np.abs(np.min(intensity))) / 2
        level = np.max(intensity) - amplitude
        sprofil, peak, sum_intensity = [[] for _ in range(3)]
        for k in range(1, 361):
            sprofil.append(amplitude * math.sin(np.deg2rad(noPeaks * k) - phase) + level)

        [peak.append(x * 360 / noPeaks) for x in range(noPeaks)]
        no_iter = 360 / noPeaks
        for k in range(int(no_iter)):
            sum_intensity.append(np.sum([sprofil[int(i)] for i in peak]))
            peak = np.add(peak, 1)

        maxpeak = np.argmax(sum_intensity)
        for k in range(noPeaks):
            peak[k] = maxpeak + k * 360 / noPeaks

        return peak, sprofil

    @staticmethod
    def determine_segmented_orientation(intensities_level, roll_marker, yaw, pitch, dirLevel_mm, CTimaging_orig):
        """ determine angles of the 6-valley artifact ('dark star') artifact around directional markers; for details cf.
        https://github.com/netstim/leaddbs/blob/master/ea_orient_main.m (lines 474f.)"""

        # Convert coordinates for directional leads to image space:
        dirLevel_vx = {k: [] for k in
                       dirLevel_mm.keys()}  # TODO: put this in the calling function as dirLevel_mm not used whatsoever
        for l, coords in dirLevel_mm.items():
            dirLevel_vx[l] = ants.transform_index_to_physical_point(CTimaging_orig,
                                                                    index=[int(x) for x in np.round(dirLevel_mm[l])])

        sum_intensities, roll_values, dir_valleys = [{k: [] for k in intensities_level.keys()} for _ in range(3)]
        roll_angles = []
        for level, intensity in intensities_level.items():
            count = 0
            shift = []

            for k in range(-30, 31):
                shift.append(k)
                rolltemp = roll_marker + np.deg2rad(k)
                dir_angles = Specific.orient_artifact_at_level(rolltemp, pitch, yaw, dirLevel_vx[level])
                temp = Specific.peaks_dir_marker(intensity, dir_angles)
                sum_intensities[level].append(temp)
                if level == list(intensities_level.keys())[0]:
                    roll_angles.append(rolltemp)
                count += 1

        dir_angles = {k: [] for k in intensities_level.keys()}
        for level in intensities_level.keys():
            temp = np.argmin(sum_intensities[level])
            roll_values[level] = roll_angles[temp]
            dir_angles[level] = Specific.orient_artifact_at_level(roll_values[level], pitch, yaw, dirLevel_vx[level])
            dir_valleys[level] = np.round(np.rad2deg(dir_angles[level]) + 1)
            dir_valleys[level][dir_valleys[level] > 360] = dir_valleys[level][dir_valleys[level] > 360] - 360
        return sum_intensities, roll_values, dir_valleys, roll_angles

    @staticmethod
    def peaks_dir_marker(intensity, angles):
        """function aiming at detecting the 'intensity peaks'; search is restricted to 360°/noPeaks. adapted from
        https://github.com/netstim/leaddbs/blob/master/ea_orient_intensitypeaksdirmarker.m """

        intensities = np.array(intensity)
        peak = np.round(np.rad2deg(angles))
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
            M, _, _, _ = Specific.orient_rollpitchyaw(roll - (deg / 60 * (2 * math.pi) / 6), pitch, yaw)
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
        dir_angles[dir_angles > 2 * math.pi] = dir_angles[dir_angles > 2 * math.pi] - 2 * math.pi
        dir_angles[dir_angles < 0] = dir_angles[dir_angles < 0] + 2 * math.pi
        dir_angles = np.sort(2 * math.pi - dir_angles)

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
