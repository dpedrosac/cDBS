#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import pickle

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import textwrap as tw
import scipy
from mat4py import loadmat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration


class PlotRoutines:
    """plots the results from the electrode reconstruction/model creation in order to validate the results"""

    def __init__(self, subject, inputfolder=''):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug=False
        self.visualise_wrapper(subject, inputfolder)

    def visualise_wrapper(self, subject, inputdir):
        """wrapper script for visualisation via pyplot from matplotlib routines"""

        filename_leadmodel = os.path.join(inputdir, 'elecModels_' + subject + '.pkl')
        lead_models, intensityProfiles, skelSkalms = self.load_leadModel(inputdir=inputdir, filename=filename_leadmodel)

        self.interactive_plot(lead_models, intensityProfiles, skelSkalms)

    def interactive_plot(self, lead_models, intensityProfiles, skelSkalms):
        """ Start plotting routine according to Lead-DBS implementation [ea_autocoord and ea_manualreconstruction]"""

        # General plotting commands
        fig = plt.figure(facecolor=self.getbgsidecolor(side=0)) # TODO: is it really the side, what is this color actually
        grid = gridspec.GridSpec(ncols=6, nrows=6, figure=fig)
        elplot = False # TODO: What this correpond to in the original code?
        lead_type = lead_models[0]['model']
        if lead_type == 'Boston Vercise Directional': # load mat-file to proceed
            mat_filename = 'boston_vercise_directed.mat'
        else:
            Output.msg_box(text="Lead type not yet implemented.", title="Lead type not implemented")
            return

        lead_properties = loadmat(os.path.join(ROOTDIR, 'ext', 'LeadDBS', mat_filename), 'r')['electrode']
        # TODO: is it necessary to save the default/pre values
        default_lead_pos = {x: np.hstack(vals) for x, vals in lead_properties.items() if x.endswith('position')}
        default_lead_coords_mm = np.array(lead_properties['coords_mm'])

        marker_orig, coords, trajectory = [[] for _ in range(3)]
        for side, _ in enumerate(lead_models):
            marker_orig.append(dict([(k, r) for k, r in lead_models[side].items() if k.startswith('marker')]))

            resize = False
            if not (lead_models[side]['first_run'] and lead_models[side]['manual_correction']):
                resize=True
                lead_models[side]['first_run'] = False

            _, lead_models[side]['trajectory'], _, marker_temp = \
                self.resolve_coordinates(marker_orig[side],default_lead_coords_mm,default_lead_pos, lead_type,
                                                                                          resize_bool=resize) # according to line 51 of ea_macor_updatescene
            # Start estimating the rotation with respect to y-Axis
            lead_models[side]['rotation'] = self.initialise_rotation(lead_models[side], marker_temp)
            xvec, yvec, lead_models[side]['rotation'], marker_rotation = self.estimate_rotation(lead_models[side],
                                                                                                marker_temp)
            lead_models[side] = self.marker_update(marker_rotation, lead_models[side])

            if xvec.size == 0 or yvec.size == 0:
                xvec, yvec, lead_models[side]['rotation'], marker_rotation = self.determine_rotation(lead_models[side],
                                                                                                 marker_rotation)
                lead_models[side] = self.marker_update(marker_rotation, lead_models[side])

            xvec_temp = np.cross(lead_models[side]["normtraj_vector"], [1, 0, 0])
            xvec_unrot = np.divide(xvec_temp, np.linalg.norm(xvec_temp))
            yvec_temp = np.cross(lead_models[side]["normtraj_vector"], [0, 1, 0])
            yvec_unrot = np.divide(yvec_temp, np.linalg.norm(yvec_temp))

            marker_temp = dict([(k, r) for k, r in lead_models[side].items() if k.startswith('marker')])
            coords_temp, traj_temp, _, _ = self.resolve_coordinates(marker_temp, default_lead_coords_mm,
                                                           default_lead_pos, lead_type, resize_bool=False) # ea_mancor_updatescene line 144
            coords.append(coords_temp)
            trajectory.append(traj_temp)
        #TODO: elecModels should be updated somehow

        #mainax2 = fig.add_subplot(grid[-1, 4:6]) # TODO: Somethings wrong herewith the grid

        emp_dist = [None]*len(lead_models)
        if lead_type == 'Boston Vercise Directional' or 'St Jude 6172' or 'St Jude 6173':
            for side, _ in enumerate(lead_models):
                coords_temp = np.zeros((4, 3))
                coords_temp[0, :] = coords[side][0, :]
                coords_temp[1, :] = np.mean(coords[side][1: 4, :], axis=0)
                coords_temp[2, :] = np.mean(coords[side][4: 7, :], axis=0)
                coords_temp[3, :] = coords[side][7, :]

                emp_dist[side] = self.lead_dist(coords=coords_temp)
        else:
            emp_dist[side] = self.lead_dist(coords=coords, factor=lead_properties['numel'])

        mean_empdist = np.mean(emp_dist)

        for side, _ in enumerate(lead_models):
            _, lead_models[side]['trajectory'], _, marker_temp = \
                self.resolve_coordinates(marker_orig[side], default_lead_coords_mm, default_lead_pos, lead_type,
                                     resize_bool=resize, rszfactor=mean_empdist)
            lead_models[side] = self.marker_update(marker_temp, lead_models[side])

        marker_new = []

        # Plot lead trajectory TODO: move to separate function
        mainax1 = fig.add_subplot(grid[:-1, 1:3])
        mainax1.axes.get_xaxis().set_visible(False)
        mainax1.axes.get_yaxis().set_visible(False)
        
        marker_plot = []
        marker_plot.append(dict([(k, r) for k, r in lead_models[0].items() if k.startswith('marker')]))

        head_marker = plt.scatter(marker_plot[0]["markers_head"][0],
                                marker_plot[0]["markers_head"][1],
                                marker_plot[0]["markers_head"][2],
                                marker='x', edgecolors='r', linewidths=1.5)

        tail_marker = plt.scatter(marker_plot[0]["markers_tail"][0],
                                marker_plot[0]["markers_tail"][1],
                                marker_plot[0]["markers_tail"][2],
                                marker='x', edgecolors='g', linewidths=1.5)

        for idx, c in enumerate(coords[0]):
            # plt.scatter(coords[0][idx][0], coords[0][idx][1], coords[0][idx][2], marker='o', edgecolor=[.9, .9, .9])
            plt.scatter(coords[0][idx][0], coords[0][idx][1], s=250, c='w', marker='p', edgecolor=[.9, .9, .9], )

        try:
            if trajectory[0].size != 0:
                traj_interp = np.linspace(start=trajectory[0][0,:], stop=trajectory[0][-1,:], num=500)
                traj_plot = plt.scatter(traj_interp[:,0], traj_interp[:,1], s = .2) #,trajectory[0][:,2],
                                        # s=200, edgecolors=[.3, .5, .9])
                # traj_plot_line = plt.plot(trajectory[0][0,:], trajectory[0][-1,:])
        except KeyError:
            print("No trajectory information available.")

        # Plot information on the right TODO: move to separate function
        text2plot = 'Lead: {} of {} \nLead spacing: {:.2f} mm\nRotation:  {} ' \
                    'deg'.format('1', str(len(lead_models)), mean_empdist, lead_models[1]['rotation'])

        fig_txt = tw.fill(tw.dedent(text2plot.rstrip()), width=80)

        # The YAxis value is -0.07 to push the text down slightly
        plt.figtext(.9, .25, text2plot, horizontalalignment='center',
                    fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='#D8D8D8',
                              ec="0.5", pad=0.5, alpha=1), fontweight='bold')

        # 'Color', 'w', 'BackgroundColor', 'k', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        # set(mcfig, 'name',[options.patientname, ', Electrode ', num2str(options.elside), '/', num2str(length(options.sides)),
        #     ', Electrode Spacing: ', sprintf('%.2f', memp_eldist), ' mm.']);


        marker_new, elplot = [],[]
        if not elplot:
            cnt = 1
            mplot1 = plt.scatter(marker_new[0]["markers_head"][0], marker_new[0]["markers_head"][1],
                                 marker_new[0]["markers_head"][2], facecolor=None,
                                 marker='*', edgecolors=[0.9,0.2,0.2])
            mplot2 = plt.scatter(marker_new[0]["markers_tail"][0], marker_new[0]["markers_tail"][1],
                                 marker_new[0]["markers_tail"][2], facecolor=None,
                                 marker='*', edgecolors=[0.2, 0.9, 0.2])
            elplot = []
            #for i in range(coords_mm.shape[0]):
            #    elplot.append(plt.scatter(coords_mm[i,0], coords_mm[i,1], coords_mm[i,2], marker='o', facecolor=None))
            #    cnt =+ 1



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
                A = np.sqrt(scipy.spatial.distance.cdist(lead_coords_mm, lead_coords_mm, 'euclidean'))  # TODO: lead_coords_mm does not correspond to electrode.coords_mm;
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

    def marker_update(self, marker_updated, lead_models):
        """replaces values of lead_models with updated values """
        for key_name, val in marker_updated.items():
            lead_models[key_name] = val
        return lead_models

    @staticmethod
    def lead_dist(coords, factor=3):
        """calculates the lead distances according to its coordinates"""

        A = scipy.spatial.distance.cdist(coords, coords, 'euclidean')
        lead_dist = np.sum(np.sum(np.tril(np.triu(A, 1), 1))) / factor
        return lead_dist

    def initialise_rotation(self, lead_models, marker):
        """script which iniitalises the estimation of a rotation; this is necessary as at the beginning there is no
        information available; This function is followed by estimate_rotation.py (see below)"""

        if lead_models['manual_correction'] and not lead_models['rotation']:
            vec_temp = marker['markers_y'] - marker['markers_head']
            vec_temp[2] = 0
            vec_temp = np.divide(vec_temp, np.linalg.norm(vec_temp))
            initial_rotation = np.degrees(math.atan2(np.linalg.norm(np.cross([0,1,0], vec_temp)),
                                                     np.dot([0,1,0], vec_temp)))
            if marker['markers_y'][0] > marker['markers_head'][0]:
                initial_rotation = - initial_rotation
            rotation = initial_rotation
        elif not lead_models['manual_correction'] and not lead_models['rotation']:
            rotation = 0

        return rotation

    def estimate_rotation(self, lead_models, marker):
        """script which determines the rotation according to the markers provided; this script includes all parts of
        the determination of rotation included in ea_mancor_updatescene of the Lead-DBS package """

        rotation = lead_models['rotation']
        normtrajvector = lead_models["normtraj_vector"]

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
    def euclidean_distance_matrix(X):
        m,n = X.shape
        D = np.zeros((m,n))
        for i in range(n):
            for j in range(i+1,n):
                diff = X[:,i] - X[:,j]
                D[i,j] = np.dot(diff, diff)
                D[j,i] = D[i,j]
        return D

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
    def save_elecModel(lead_models, intensityProfiles, skelSkalms, filename=''):

        if not filename:
            Output.msg_box(text='No filename for saving lead model provided', title='No filename provided')
            return

        with open(filename, "wb") as f:
            pickle.dump(lead_models, f)
            pickle.dump(intensityProfiles, f)
            pickle.dump(skelSkalms, f)

