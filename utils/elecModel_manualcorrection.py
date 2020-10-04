#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import scipy
from mat4py import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration


class PlotRoutines:
    """plots the results from the electrode reconstruction/model creation in order to validate the results"""

    def __init__(self, subject, inputfolder=''):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug=False

        filename_leadmodel = os.path.join(inputfolder, 'elecModels_' + subject + '.pkl')
        lead_models, intensityProfiles, skelSkalms = self.load_leadModel(inputfolder, filename=filename_leadmodel)
        self.interactive_plot(lead_models[1], intensityProfiles, skelSkalms)

    def interactive_plot(self, lead_model, intensityProfiles, skelSkalms):
        """ Start plotting routine according to Lead-DBS implementation [ea_autocoord and ea_manualreconstruction]"""

        lead_properties, default_positions, default_coordinates = GetData.get_default_lead(lead_model) #all in [mm]
        default_lead_pos = {x: np.hstack(vals) for x, vals in lead_properties.items() if x.endswith('position')}
        default_lead_coords_mm = np.array(lead_properties['coords_mm'])

        marker, coordinates, trajectory, unrot, resize = GetData.get_leadModelProperties(self, lead_model)

        #TODO: elecModels should be updated somehow

        #mainax2 = fig.add_subplot(grid[-1, 4:6]) # TODO: Somethings wrong herewith the grid

        if lead_model['model'] == 'Boston Vercise Directional' or 'St Jude 6172' or 'St Jude 6173':
            coords_temp = np.zeros((4, 3))
            coords_temp[0, :] = coordinates[0, :]
            coords_temp[1, :] = np.mean(coordinates[1: 4, :], axis=0)
            coords_temp[2, :] = np.mean(coordinates[4: 7, :], axis=0)
            coords_temp[3, :] = coordinates[7, :]

            emp_dist = self.lead_dist(coords=coords_temp)
        else:
            emp_dist = self.lead_dist(coords=coordinates, factor=lead_properties['numel'])

        mean_empdist = np.mean(emp_dist)

        _, lead_model['trajectory'], _, marker_temp = \
            self.resolve_coordinates(marker, default_lead_coords_mm, default_lead_pos, lead_model['model'],
                                     resize_bool=resize, rszfactor=mean_empdist)
        lead_model = self.marker_update(marker_temp, lead_model)

        # Start plotting
        fig = plt.figure(facecolor=self.getbgsidecolor(side=0)) # TODO: is it really the side, what is this color actually
        fig.tight_layout()
        width = [1, 2, 2, 1]
        height = [1, 1, 1]
        grid = gridspec.GridSpec(ncols=4, nrows=3, figure=fig, width_ratios=width, height_ratios=height)

        grid_indices = [(0,5), (1,3)]
        self.plotCTintensities(lead_model, coordinates, trajectory, fig, grid, grid_indices)

        grid_indices = [(-1), (-1)]
        self.plot_leadInformation(lead_model, mean_empdist, fig, grid, grid_indices)

        self.plotCTaxial(lead_model, fig, grid, cmap='gist_gray')
        self.plot_leadModel(lead_properties, fig, grid)

    def plotCTintensities(self, lead_model, coordinates, trajectory, fig, grid, grid_indices, dimension=['sag', 'cor']):
        """function plotting perpendicular images of the intensities obtained from trajectory coordinates; separate
        function needed here as changes occur in trejectories/markers"""

        dimension = ['sag', 'cor'] # TODO redundant, remove when debugging is finished!
        print("\t...extracting intensities for corresponding CTimaging\t...", end='')
        intensity_matrix, bounding_box, fitvolume = self.get_xyplanes(trajectory=trajectory, lead_model=lead_model,
                                                          direction=dimension)
        print("Done!", flush=True)
        # Prepare plot
        mainax1 = fig.add_subplot(grid[grid_indices[0][0]:grid_indices[0][1],
                                  grid_indices[1][0]:grid_indices[1][1]], facecolor='None', projection='3d')
        fig.tight_layout()
        mainax1.set_axis_off()
        mainax1.set_facecolor('None')

        face = {x: [] for x in intensity_matrix.keys()}
        for orientation, intensity in intensity_matrix.items():
            matrix2plot = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity)) # Prepare data!
            fcolors = GetData.color_plotsurface(matrix2plot, cmap='gist_gray', alpha=.2)
            X, Y, Z = fitvolume[orientation][0, :, :], fitvolume[orientation][1, :, :], fitvolume[orientation][2, :, :]
            face[orientation] = mainax1.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False, facecolors=fcolors,
                                                     alpha=.1,antialiased=True).set_facecolor((0, 0, 1, .3))
            mainax1.view_init(elev=0, azim=180)
            mainax1.autoscale(tight=True)

            print(orientation)
        mainax1.view_init(elev=0, azim=180)
        grid.tight_layout(fig)
        plt.draw()

        traj_interp = np.linspace(start=trajectory[0, :], stop=trajectory[-1, :], num=500)
        mainax1.plot([traj_interp[0,0], traj_interp[-1,0]], [traj_interp[0,1], traj_interp[-1,1]],
                    zs=[traj_interp[0,2], traj_interp[-1,2]], color='magenta')

        for c in coordinates:
            self.plot_coordinates(c, mainax1, marker='p', s=250)

        # Start plotting the markers for the 'head' and 'tail' of the lead
        marker_plot = dict([(k, r) for k, r in lead_model.items() if k.startswith('marker')])
        self.plot_coordinates(marker_plot["markers_head"], mainax1, marker='o', facecolor='g', edgecolor='g',
                              linewidth=1.5)
        self.plot_coordinates(marker_plot["markers_tail"], mainax1, marker='o', facecolor='r', edgecolor='r',
                              linewidth=1.5)
        mainax1.set_zlim(np.multiply(mainax1.get_zlim(),[1, .98]))

    @staticmethod
    def plot_coordinates(coordinates, axis, marker='', s=25, facecolor='k', linewidth=1.5,edgecolor=[.9]*3):
        """plots all coordinates of the electrodes estimated from the lead model"""

        axis.scatter(coordinates[0], coordinates[1], coordinates[2], s=s, c=facecolor, edgecolor=edgecolor,
                     marker=marker, linewidth=linewidth)

    @staticmethod
    def plot_leadInformation(lead_model, mean_empdist, fig, grid, grid_indices):
        """plots lead information in the lower right corner"""

        axis = fig.add_subplot(grid[grid_indices[0], grid_indices[1]], facecolor='None')
        fig.tight_layout()
        axis.set_axis_off()
        axis.set_facecolor('None')

        stn_direction = 'right' if not lead_model['trajectory'][0, 0] > lead_model['trajectory'][-1, 0] else 'left'
        text2plot = 'Lead: {} STN \nLead spacing: {:.2f} mm\nRotation:  {} ' \
                    'deg'.format(stn_direction, mean_empdist, lead_model['rotation'])
        axis.text(.6, .25, text2plot, horizontalalignment='center',fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='#D8D8D8', ec="0.5", pad=0.5, alpha=1), fontweight='bold')

    def plotCTaxial(self, lead_model, fig, grid, cmap='gist_gray'):
        """function plotting axial sclices at the level of head and tail markers"""

        print("\t...extracting axial slices for corresponding markers of CTimaging\t...")
        marker_of_interest = ['markers_head', 'markers_tail']
        marker_plot = dict([(k, r) for k, r in lead_model.items() if k.startswith('marker') and
                            any(z in k for z in marker_of_interest)])
        color_specs = ['g', 'r']
        item = -1
        for marker, coordinates in marker_plot.items():
            item += 1
            intensity_matrix, bounding_box, _ = self.get_axialplanes(coordinates, lead_model=lead_model,
                                                          window_size=15, resolution=.5)
            transversal_axis = fig.add_subplot(grid[item, -1], facecolor='None')
            transversal_axis.imshow(intensity_matrix, cmap=cmap, extent=[np.min(bounding_box, axis=1)[0],
                                                                         np.max(bounding_box, axis=1)[0],
                                                                         np.min(bounding_box, axis=1)[1],
                                                                         np.max(bounding_box, axis=1)[1]],
                                    interpolation='bicubic')
            transversal_axis.set_axis_off()
            transversal_axis.set_facecolor('None')

            transversal_axis.scatter(coordinates[0], coordinates[1], s=200, c=color_specs[item],
                                     edgecolor=color_specs[item], marker='x', linewidth=1.5)

        grid.tight_layout(fig)

    @staticmethod
    def plot_leadModel(lead_properties, fig, grid, grid_indices=[(0, -1), (0)],
                       colors=[.3, .5], items_of_interest=['insulation', 'contacts']):
        """plots a schematic lead model at the left side in order to visualise what it should look like """

        lead_model_plot = fig.add_subplot(grid[grid_indices[0][0]:grid_indices[0][1],
                                  grid_indices[1]], facecolor='None', projection='3d')

        # Prepare data
        lead_plot = dict([(k, r) for k, r in lead_properties.items() if any(z in k for z in items_of_interest)])
        max_coords, min_coords = [[] for _ in range(2)]

        # Start visualisation
        for idx, item in enumerate(items_of_interest):
            for idx_vertices in (range(0, len(lead_plot[item]['vertices']))):
                mesh = []
                verts_temp = np.array(lead_plot[item]['vertices'][idx_vertices]) - 1
                faces_temp = np.array(lead_plot[item]['faces'][idx_vertices]) - 1
                mesh = Poly3DCollection(verts_temp[faces_temp], facecolors=[colors[idx]] * 3, edgecolor="none",
                                        rasterized=True)
                lead_model_plot.add_collection(mesh)

                if item == 'contacts':
                    max_coords.append(np.max(verts_temp[faces_temp], axis=1))
                    min_coords.append(np.min(verts_temp[faces_temp], axis=1))

        lead_type = 'boston'
        marker_position = ['head', 'tail']
        marker = {k: [] for k in marker_position}
        if lead_type == 'boston':
            for n in range(3):
                marker['tail'].append(lead_properties['tail_position'][n])
                marker['head'].append(lead_properties['tail_position'][n])
            marker['tail'][2] = np.min(min_coords[0])

        else:
            for n in range(3):
                marker['tail'].append(lead_properties['tail_position'][n])
                marker['head'].append(lead_properties['head_position'][n])

        color = ['r', 'g']
        for idx, loc in enumerate(marker_position):
            lead_model_plot.scatter(marker[loc][0] - 1, marker[loc][1] - 2, marker[loc][2],
                       c=color[idx], s=25)

        lead_model_plot.set_zlim([0, 15])
        lead_model_plot.set_ylim(-4, 4)
        lead_model_plot.set_xlim(-4, 4)
        lead_model_plot.set_facecolor('None')
        lead_model_plot.view_init(elev=0, azim=-120)
        lead_model_plot.set_axis_off()
        grid.tight_layout(fig)

    def resolve_coordinates(self, marker, lead_coords_mm, lead_positions, lead_type, resize_bool=False, rszfactor=0):
        """emulates the function from Lead-DBS ea_resolvecoords; unlike in Lead DBS this is done one at a time cf.
        https://github.com/netstim/leaddbs/blob/master/templates/electrode_models/ea_resolvecoords.m"""

        if resize_bool:
            can_dist = np.linalg.norm(lead_positions["head_position"] - lead_positions["tail_position"])

            if lead_type == 'Boston Vercise Directional' or 'St Jude 6172' or 'St Jude 6173':
                coords_temp = np.zeros((4,3))
                coords_temp[0,:] = lead_coords_mm[0,:]
                coords_temp[1,:] = np.mean(lead_coords_mm[1: 4,:], axis=0)
                coords_temp[2,:] = np.mean(lead_coords_mm[4: 7,:], axis=0)
                coords_temp[3, :] = lead_coords_mm[7, :]

                A = scipy.spatial.distance.cdist(coords_temp, coords_temp, 'euclidean') # move this part into helper function
                can_eldist = np.sum(np.sum(np.tril(np.triu(A,1),1)))/ 3
            else:
                A = np.sqrt(scipy.spatial.distance.cdist(lead_coords_mm, lead_coords_mm, 'euclidean'))
                can_eldist = np.sum(np.sum(np.tril(np.triu(A, 1), 1))) / 3 # TODO what is (options.elspec.numel -1)?? change code after finding out!!

            if rszfactor != 0:
                stretch = can_dist * (rszfactor / can_eldist)
            else:
                stretch = can_dist

            vec = np.divide((marker["markers_tail"] - marker["markers_head"]),
                            np.linalg.norm(marker["markers_tail"] - marker["markers_head"]))
            marker["markers_tail"] = marker["markers_head"] + vec * stretch

        coords, traj_vector, trajectory, can_eldist = [[] for _ in range(4)]
        if not marker["markers_head"].size==0:
            M = np.stack((np.append(marker["markers_head"], 1), np.append(marker["markers_tail"], 1),
                          np.append(marker["markers_x"], 1), np.append(marker["markers_y"], 1)))
            E = np.stack((np.append(lead_positions["head_position"], 1), np.append(lead_positions["tail_position"], 1),
                          np.append(lead_positions["x_position"], 1), np.append(lead_positions["y_position"], 1)))

            X = np.linalg.lstsq(E, M, rcond=None)

            coords_mm = np.concatenate([lead_coords_mm, np.ones(shape=(lead_coords_mm.shape[0],1))], axis=1)
            coords = (coords_mm @ X[0]).T
            coords = coords[0: 3,:].T

            traj_vector = (marker["markers_tail"] - marker["markers_head"]) / \
                          np.linalg.norm(marker["markers_tail"] - marker["markers_head"])

            trajectory = np.stack((marker["markers_head"] - traj_vector*5,
                                   marker["markers_head"] + traj_vector*25))
            trajectory = np.array((np.linspace(trajectory[0, 0], trajectory[1, 0], num=50),
                                   np.linspace(trajectory[0, 1], trajectory[1, 1], num=50),
                                   np.linspace(trajectory[0, 2], trajectory[1, 2], num=50))).T

        return coords, trajectory, can_eldist, marker

    # ====================    Modules related to the estimation of rotation   ====================
    @staticmethod
    def initialise_rotation(lead_model, marker):
        """script iniitalising the estimation of rotation angles; necessary as at the beginning there is no
        information available; This function is followed by estimate_rotation.py (see below)"""

        if lead_model['manual_correction'] and not lead_model['rotation']:
            vec_temp = marker['markers_y'] - marker['markers_head']
            vec_temp[2] = 0
            vec_temp = np.divide(vec_temp, np.linalg.norm(vec_temp))
            initial_rotation = np.degrees(math.atan2(np.linalg.norm(np.cross([0,1,0], vec_temp)),
                                                     np.dot([0,1,0], vec_temp)))
            if marker['markers_y'][0] > marker['markers_head'][0]:
                initial_rotation = - initial_rotation
            rotation = initial_rotation
        elif not lead_model['manual_correction'] and not lead_model['rotation']:
            rotation = 0

        return rotation

    @staticmethod
    def estimate_rotation(lead_models, marker):
        """determination of rotation according to markers provided; follows steps in ea_mancor_updatescene of
         Lead-DBS package (cf. https://github.com/ningfei/lead/blob/develop/ea_mancor_updatescene.m) """

        rotation, normtrajvector = lead_models['rotation'], lead_models["normtraj_vector"]
        # normtrajvector = lead_models["normtraj_vector"] # TODO remove if no error is dropped

        yvec = np.zeros((3,1))
        yvec[0] = -np.cos(0) * np.sin(np.deg2rad(rotation))
        yvec[1] = (np.cos(0) * np.cos(np.deg2rad(rotation))) + (np.sin(0) * np.sin(np.deg2rad(rotation)) * np.sin(0))
        yvec[2] = (-np.sin(0) * np.cos(np.deg2rad(rotation))) + (np.cos(0) * np.sin(np.deg2rad(rotation)) * np.sin(0))

        xvec = np.cross(yvec.T, [0,0,1])
        xvec = xvec - (np.dot(xvec, normtrajvector) / np.linalg.norm(normtrajvector)**2) * normtrajvector
        xvec = np.divide(xvec, np.linalg.norm(xvec))
        yvec = -np.cross(xvec, normtrajvector)

        marker['markers_x'] = marker['markers_head'] + (xvec * lead_models['lead_diameter'] / 2)
        marker['markers_y'] = marker['markers_head'] + (yvec * lead_models['lead_diameter'] / 2)

        return xvec, yvec, rotation, marker


    def get_xyplanes(self, trajectory, lead_model, limits=[(-4,4),(-4,4),(-10,20)], sample_width=10,
                     direction=['sag', 'cor']):

        hd_trajectories = self.interpolate_trajectory(trajectory, resolution=10) # TODO: rename traj to trajectory after assigning in function

        slices = {k: [] for k in direction}
        imat = {k: [] for k in direction}
        fitvolume = {k: [] for k in direction}
        bounding_box = {k: [] for k in direction}

        for idx, plane in enumerate(direction):
            slices[plane] = list(range(limits[idx][0], limits[idx][1]+1, 1))
            imat[plane], fitvolume[plane] = self.resample_CTplanes(hd_trajectories, plane, lead_model, resolution=.35)

            if plane is not 'sag':
                span_vector = [sample_width, 0, 0]
            else:
                span_vector = [0, sample_width, 0]

            idx = [0, -1]
            bounding_box_coords = []
            for k in idx:
                bounding_box_coords.append(hd_trajectories[k,:]-span_vector)
                bounding_box_coords.append(hd_trajectories[k, :] + span_vector)
            bounding_box_coords = np.array(bounding_box_coords)

            axes_name = ['xx', 'yy', 'zz']
            box = {k: [] for k in axes_name}
            for i, dim in enumerate(axes_name):
                box[dim] = bounding_box_coords.T[i,:].tolist()

            bounding_box[plane] = box

        return imat, bounding_box, fitvolume

    def get_axialplanes(self, marker_coordinates, lead_model, window_size=15, resolution=.5):
        """returns a plane at a specific window with a certain direction"""

        if lead_model['transformation_matrix'].shape[0] == 3:
            lead_model['transformation_matrix'] = np.eye(4)*lead_model['transformation_matrix'][0,0]
            lead_model['transformation_matrix'][-1,-1] = 1
        transformation_matrix = lead_model['transformation_matrix']
        transformation_matrix = np.eye(4)

        bounding_box_coords = []
        for k in range(2):
            bounding_box_coords.append(np.arange(start=marker_coordinates[k]-window_size,
                                                     stop=marker_coordinates[k]+window_size, step=resolution))
        bounding_box_coords.append(np.repeat(marker_coordinates[-1], len(bounding_box_coords[1])))
        bounding_box = np.array(bounding_box_coords)

        meshX, meshY = np.meshgrid(bounding_box[0,:], bounding_box[1,:])
        meshZ = np.repeat(bounding_box[-1,0], len(meshX.flatten()))
        fitvolume_orig = np.array([meshX.flatten(), meshY.flatten(), meshZ.flatten(), np.ones(meshX.flatten().shape)])
        fitvolume = np.linalg.solve(transformation_matrix, fitvolume_orig)
        resampled_points = PlotRoutines.interpolate_CTintensities(lead_model, fitvolume)
        imat = np.reshape(resampled_points, (meshX.shape[0], -1), order='F')

        return imat, bounding_box, fitvolume

    @staticmethod
    def resample_CTplanes(hd_trajectories, direction, lead_model, resolution=.2, sample_width=10):
        """This function resamples the intesities of the source imaging to a grid which is later used to visualise the
        leads. ea_mancor_updatescene lines 264f."""

        if type(direction) == list:
            direction = ''.join(direction)

        if lead_model['transformation_matrix'].shape[0] == 3:
            lead_model['transformation_matrix'] = np.eye(4)*lead_model['transformation_matrix'][0,0]
            lead_model['transformation_matrix'][-1,-1] = 1
        transformation_matrix = lead_model['transformation_matrix']
        transformation_matrix = np.eye(4)

        xvec = np.arange(start=-sample_width, stop=sample_width+resolution, step=resolution)
        meanfitline = np.vstack((hd_trajectories.T, np.ones(shape=(1, hd_trajectories.T.shape[1])))) # needed for transformation
        addvolume = np.tile(xvec,(len(meanfitline.T),1))

        fitvolume = []
        for t in range(4):
            fitvolume.append(np.tile(meanfitline[t,:], xvec.shape).reshape(xvec.shape[0], meanfitline.shape[1]).T)
        fitvolume_orig = np.stack(fitvolume)

        if direction is 'cor':
            fitvolume_orig[0,:,:] += addvolume
        elif direction is 'sag':
            fitvolume_orig[1, :, :] += addvolume
        elif direction is 'tra':
            fitvolume_orig[2, :, :] += addvolume

        fitvolume = np.linalg.solve(transformation_matrix, np.reshape(fitvolume_orig, (4, -1), order='F'))
        #fitvolume[[2, 1, 0, 3]] = fitvolume[[0, 1, 2, 3]] # necessary for sitk?!?
        resampled_points = PlotRoutines.interpolate_CTintensities(lead_model, fitvolume)
        imat = np.reshape(resampled_points, (meanfitline.shape[1], -1), order='F')

        return imat, fitvolume_orig

    # ========================================    Interpolations   ========================================
    @staticmethod
    def interpolate_trajectory(orig_trajectory, resolution=20):
        """interpolates between trajectory points thus creating a „high resolution“ version of it"""

        hd_trajectory = []
        for idx in range(np.array(orig_trajectory).shape[1]):
            f = scipy.interpolate.interp1d(np.linspace(start=1, stop=50), np.array(orig_trajectory)[:, idx])
            hd_trajectory.append(f(np.linspace(start=1, stop=50, num=(-1 + len(orig_trajectory[:, idx]))
                                                                     * resolution + 1)))
        return np.stack(hd_trajectory).T

    def interpolate_CTintensities(lead_model, fitvolume):
        import SimpleITK as sitk

        img = sitk.ReadImage(os.path.join(*lead_model['filenameCTimaging']))
        physical_points = list(map(tuple, fitvolume[0:3,:].T))
        #physical_points = physical_points[0:5]
        num_samples = len(physical_points)
        physical_points = [img.TransformContinuousIndexToPhysicalPoint(pnt) for pnt in physical_points]

        #interp_grid_img = sitk.Image((len(physical_points) *([1] * (img.GetDimension() - 1))), sitk.sitkUInt8)
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

        resampled_points = [resampled_temp[x,0,0] for x in range(resampled_temp.GetWidth())]
        debug = False
        if debug:
            for i in range(resampled_temp.GetWidth()):
                print(str(img.TransformPhysicalPointToContinuousIndex(physical_points[i])) + ': ' + str(resampled_temp[[i] + [0] * (img.GetDimension() - 1)]) + '\n')

        return np.array(resampled_points)

    # ====================    General Helper Functions for manual correction   ====================
    @staticmethod
    def get_default_lead(lead_model):
        """obtains the default lead properties according to the model """
        #TODO: create a new helperfunctions class and move this part there

        if lead_model['model'] == 'Boston Vercise Directional': # load mat-file to proceed
            mat_filename = 'boston_vercise_directed.mat'
            lead_properties = loadmat(os.path.join(ROOTDIR, 'ext', 'LeadDBS', mat_filename), 'r')['electrode']
            default_positions = {x: np.hstack(vals) for x, vals in lead_properties.items() if x.endswith('position')}
            default_mm_coordinates = np.array(lead_properties['coords_mm'])
        else:
            Output.msg_box(text="Lead type not yet implemented.", title="Lead type not implemented")
            return

        return lead_properties, default_positions, default_mm_coordinates

    @staticmethod
    def load_leadModel(inputdir, filename):
        """Function aiming at loading the lead models saved in preprocLead.py from this toolbox"""

        if not inputdir:
            Output.msg_box(text="No input folder provided, please double-check!", title="Missing input folder")
            return
        elif not os.path.isfile(filename):
            Output.msg_box(text="Models for electrode not available, please run detection first!",
                           title="Models not available")
            return
        else:
            with open(filename, "rb") as model: # roughly ea_loadreconstruction in the LeadDBS script
                lead_models = pickle.load(model)
                intensityProfiles = pickle.load(model)
                skelSkalms = pickle.load(model)

        return lead_models, intensityProfiles, skelSkalms

    @staticmethod
    def save_leadModel(lead_models, intensityProfiles, skelSkalms, filename=''):

        if not filename:
            Output.msg_box(text='No filename for saving lead model provided', title='No filename provided')
            return

        with open(filename, "wb") as f:
            pickle.dump(lead_models, f)
            pickle.dump(intensityProfiles, f)
            pickle.dump(skelSkalms, f)

    def marker_update(self, marker_updated, lead_models):
        """replaces values of lead_models with updated values """

        for key_name, val in marker_updated.items():
            lead_models[key_name] = val
        return lead_models

    @staticmethod
    def lead_dist(coords, factor=3):
        """calculate lead distances according to its coordinates"""

        A = scipy.spatial.distance.cdist(coords, coords, 'euclidean')
        lead_dist = np.sum(np.sum(np.tril(np.triu(A, 1), 1))) / factor

        return lead_dist


    # ====================    Helper Functions fpr plotting Data   ====================

    @staticmethod
    def getbgsidecolor(side, xray=False):
        """ """
        from matplotlib import colors

        line_cols = matplotlib.cm.get_cmap('Set1', 64) # TODO: maybe a cmap would make sense
        line_cols = colors.rgb_to_hsv(line_cols(np.linspace(0, 1, 64))[:,0:3])
        line_cols[:,-1] = line_cols[:,-1]/3
        if xray:
            line_cols[:, 1] = line_cols[:, 1] / 1.5
            line_cols[:, 2] = line_cols[:, 2] * 1.5

        line_cols = colors.hsv_to_rgb(line_cols)
        col = line_cols[side,:] # TODO: why on earth is this so complicated to get these colors

        return col

class GetData:
    def __init__(self, parent=PlotRoutines):
        self.parent = parent

    def get_leadModelProperties(self, lead_model, resize=False):
        """ reads and estimates all necessary markers and trajectories for the corresponding lead model """
        print("\t... reading properties of the lead model and estimating rotation")

        lead_properties, default_positions, default_coordinates = GetData.get_default_lead(lead_model) #all in [mm]
        marker_unprocessed = dict([(k, r) for k, r in lead_model.items() if k.startswith('marker')])

        if not (lead_model['first_run'] and lead_model['manual_correction']):
            resize=True #TODO There is no consequence resulting from this option
            lead_model['first_run'] = False

        _, lead_model['trajectory'], _, marker_temp = \
            self.resolve_coordinates(marker_unprocessed, default_coordinates, default_positions, lead_model['model'],
                                         resize_bool=resize)
        # lead_model['rotation'] = self.initialise_rotation(lead_model, marker_temp)
        lead_model['rotation'] = GetData.initialise_rotation(lead_model, marker_temp)
        xvec, yvec, lead_model['rotation'], marker_rotation = self.estimate_rotation(lead_model, marker_temp)
        lead_model = self.marker_update(marker_rotation, lead_model)

        if xvec.size == 0 or yvec.size == 0:
            xvec, yvec, lead_model['rotation'], marker_rotation = self.estimate_rotation(lead_model, marker_rotation)
            lead_model = self.marker_update(marker_rotation, lead_model)

        options2process = {'xvec': [1, 0, 0], 'yvec': [0, 1, 0]}
        unrot = {k: [] for k in options2process.keys()}
        for key in options2process:
            vec_temp = np.cross(lead_model["normtraj_vector"], options2process[key])
            unrot[key] = np.divide(vec_temp, np.linalg.norm(vec_temp))

        marker = dict([(k, r) for k, r in lead_model.items() if k.startswith('marker')])
        coordinates, trajectory, _, _ = self.resolve_coordinates(marker, default_coordinates,
                                                           default_positions, lead_model['model'], resize_bool=False) # ea_mancor_updatescene line 144
        return marker, coordinates, trajectory, unrot, resize

    @staticmethod
    def get_default_lead(lead_model):
        """obtains the default lead properties according to the model """
        #TODO: create a new helperfunctions class and move this part there

        if lead_model['model'] == 'Boston Vercise Directional': # load mat-file to proceed
            mat_filename = 'boston_vercise_directed.mat'
            lead_properties = loadmat(os.path.join(ROOTDIR, 'ext', 'LeadDBS', mat_filename), 'r')['electrode']
            default_positions = {x: np.hstack(vals) for x, vals in lead_properties.items() if x.endswith('position')}
            default_mm_coordinates = np.array(lead_properties['coords_mm'])
        else:
            Output.msg_box(text="Lead type not yet implemented.", title="Lead type not implemented")
            return

        return lead_properties, default_positions, default_mm_coordinates

    # ====================    Modules related to the estimation of rotation   ====================
    @staticmethod
    def initialise_rotation(lead_model, marker):
        """script iniitalising the estimation of rotation angles; necessary as at the beginning there is no
        information available; This function is followed by estimate_rotation.py (see below)"""

        if lead_model['manual_correction'] and not lead_model['rotation']:
            vec_temp = marker['markers_y'] - marker['markers_head']
            vec_temp[2] = 0
            vec_temp = np.divide(vec_temp, np.linalg.norm(vec_temp))
            initial_rotation = np.degrees(math.atan2(np.linalg.norm(np.cross([0,1,0], vec_temp)),
                                                     np.dot([0,1,0], vec_temp)))
            if marker['markers_y'][0] > marker['markers_head'][0]:
                initial_rotation = - initial_rotation
            rotation = initial_rotation
        elif not lead_model['manual_correction'] and not lead_model['rotation']:
            rotation = 0

        return rotation

    @staticmethod
    def estimate_rotation(lead_models, marker):
        """determination of rotation according to markers provided; follows steps in ea_mancor_updatescene of
         Lead-DBS package (cf. https://github.com/ningfei/lead/blob/develop/ea_mancor_updatescene.m) """

        rotation, normtrajvector = lead_models['rotation'], lead_models["normtraj_vector"]
        # normtrajvector = lead_models["normtraj_vector"] # TODO remove if no error is dropped

        yvec = np.zeros((3,1))
        yvec[0] = -np.cos(0) * np.sin(np.deg2rad(rotation))
        yvec[1] = (np.cos(0) * np.cos(np.deg2rad(rotation))) + (np.sin(0) * np.sin(np.deg2rad(rotation)) * np.sin(0))
        yvec[2] = (-np.sin(0) * np.cos(np.deg2rad(rotation))) + (np.cos(0) * np.sin(np.deg2rad(rotation)) * np.sin(0))

        xvec = np.cross(yvec.T, [0,0,1])
        xvec = xvec - (np.dot(xvec, normtrajvector) / np.linalg.norm(normtrajvector)**2) * normtrajvector
        xvec = np.divide(xvec, np.linalg.norm(xvec))
        yvec = -np.cross(xvec, normtrajvector)

        marker['markers_x'] = marker['markers_head'] + (xvec * lead_models['lead_diameter'] / 2)
        marker['markers_y'] = marker['markers_head'] + (yvec * lead_models['lead_diameter'] / 2)

        return xvec, yvec, rotation, marker

    @staticmethod
    def color_plotsurface(imat2plot, cmap='gist_gray', alpha=1):
        """defines a colormap which is used for plotting data later """
        import matplotlib.cm as cm
        from matplotlib import colors as colors

        color_dimension = imat2plot  # change to desired fourth dimension
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)
        fcolors[:,:,-1] = np.ones((fcolors[:,:,-1].shape))*alpha

        return fcolors