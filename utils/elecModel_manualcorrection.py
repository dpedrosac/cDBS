#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt

from dependencies import ROOTDIR
from utils.HelperFunctions import Output, Configuration


class PlotRoutines:
    """plots the results from the electrode reconstruction/model creation in order to validate the results"""

    def __init__(self, subject, inputfolder=''):
        self.cfg = Configuration.load_config(ROOTDIR)
        self.debug=False
        self.visualise_wrapper(subject, inputfolder)

    def visualise_wrapper(self, subject, inputfolder):
        """wrapper script for visualisation via pyplot from matplotlib routines"""

        filename_elecmodel = os.path.join(inputfolder, 'elecModels_' + subject + '.pkl')
        if not inputfolder:
            Output.msg_box(text="No input folder provided, please double-check!")
            return
        elif not os.path.isfile(filename_elecmodel):
            Output.msg_box(text="models for electrode not available, please run detection first!")
            return
        else:
            with open(filename_elecmodel, "rb") as model: # roughly ea_loadreconstruction in the LeadDBS script
                elecModels = pickle.load(model)
                intensityProfiles = pickle.load(model)
                skelSkalms = pickle.load(model)

        self.interactive_plot(elecModels, intensityProfiles, skelSkalms)

    def interactive_plot(self, elecModels, intensityProfiles, skelSkalms):
        """ start plotting routine according to Lead-DBS implementation [ea_autocoord and ea_manualreconstruction]"""

        fig = plt.figure(facecolor=self.getbgsidecolor(side=0)) # TODO: is it really the side, what is this color actually
        grid = gridspec.GridSpec(ncols=3, nrows=6, figure=fig)

        elplot = False

        lead_type = elecModels[0]['model']
        if lead_type == 'Boston Vercise Directional':
            mat_filename = 'boston_vercise_directed.mat'
        else:
            Output.msg_box("Not yet implemented")
            return

        lead_properties = h5py.File(os.path.join(ROOTDIR, 'ext', 'LeadDBS', mat_filename), 'r')['electrode']
        default_lead_pos = {x: np.hstack(vals) for x, vals in lead_properties.items() if x.endswith('position')}
        default_lead_coords_mm = np.array(lead_properties['coords_mm']).T

        markers_orig, markers, trajectory, lead_dist = [[] for _ in range(4)]
        for idx, e in enumerate(elecModels):
            markers_orig.append(dict([(k, r) for k, r in e.items() if k.startswith('marker')]))

            if not (e["first_run"] & e["manual_correction"]):
                resize=True
            else: resize=False

            _, trajectory_temp, markers_temp, _ = self.resolve_coordinates(markers_orig, default_lead_coords_mm,
                                                                     default_lead_pos, lead_type, resize=resize)
            trajectory.append(trajectory_temp)

            xvec, yvec, markers_temp, normtraj_vector = self.determine_rotation(e, markers_temp)

            xvec_unrot = np.cross(normtraj_vector, [1, 0, 0])
            xvec_unrot = np.divide(xvec_unrot, np.linalg.norm(xvec_unrot))
            yvec_unrot = np.cross(normtraj_vector, [0, 1, 0])
            yvec_unrot = np.divide(yvec_unrot, np.linalg.norm(yvec_unrot))

            markers.append(markers_temp)
            _, _, _, can_eldist = self.resolve_coordinates(markers_orig, default_lead_coords_mm,
                                                       default_lead_pos, lead_type, resize=resize)
            lead_dist.append(can_eldist)

        _, trajectory_temp, markers_temp, _ = self.resolve_coordinates(markers_orig, default_lead_coords_mm, default_lead_pos,
                                                       lead_type, resize=resize, rszfactor=np.mean(can_eldist))

        mainax1 = fig.add_subplot(grid[:-1, 1:3])
        mainax2 = fig.add_subplot(grid[:-1, 4:6])

        if not elplot:
            cnt = 1
            mplot1 = plt.scatter(markers[0]["markers_head"][0], markers[0]["markers_head"][1],
                                 markers[0]["markers_head"][2], facecolor=None,
                                 marker='*', edgecolors=[0.9,0.2,0.2])
            mplot2 = plt.scatter(markers[0]["markers_tail"][0], markers[0]["markers_tail"][1],
                                 markers[0]["markers_tail"][2], facecolor=None,
                                 marker='*', edgecolors=[0.2, 0.9, 0.2])
            elplot = []
            #for i in range(coords_mm.shape[0]):
            #    elplot.append(plt.scatter(coords_mm[i,0], coords_mm[i,1], coords_mm[i,2], marker='o', facecolor=None))
            #    cnt =+ 1



    def resolve_coordinates(self, markers, lead_coords_mm, lead_positions, lead_type, resize=False, rszfactor=0):
        """emulates the function from Lead-DBS ea_resolvecoords cf.
        https://github.com/netstim/leaddbs/blob/master/templates/electrode_models/ea_resolvecoords.m"""
        from sklearn.metrics.pairwise import euclidean_distances

        if resize:
            can_dist = np.linalg.norm(lead_positions["head_position"] - lead_positions["tail_position"])

            if lead_type == 'Boston Vercise Directional' or 'STJ-6172' or 'STJ-6173':
                coords_temp = np.zeros((4,3))
                coords_temp[0,:] = lead_coords_mm[0,:]
                coords_temp[1,:] = np.mean(lead_coords_mm[1: 4,:], axis=0)
                coords_temp[2,:] = np.mean(lead_coords_mm[4: 7,:], axis=0)
                coords_temp[3, :] = lead_coords_mm[7, :]

                A = self.euclidean_distance_matrix(coords_temp)
                can_eldist = np.sum(np.sum(np.tril(np.triu(A,1),1)))/ 3
            else:
                A = np.sqrt(euclidean_distances(lead_coords_mm, lead_coords_mm))  # TODO: Does this correspond to sqrt(ea_sqdist(lead_coords_mm', lead_coords_mm'));
                can_eldist = np.sum(np.sum(np.tril(np.triu(A, 1), 1))) / 3 # TODO what is (options.elspec.numel -1)?? change code after finding out!!

            if rszfactor != 0:
                stretch = can_dist * (rszfactor / can_eldist)
            else:
                stretch = can_dist

            for idx in enumerate(markers):
                vec = (markers[idx]["markers_tail"] - markers[idx]["markers_head"]) / \
                      np.linalg.norm(markers[idx]["markers_tail"] - markers[idx]["markers_head"])
                markers[idx]["markers_tail"] = markers[idx]["markers_head"] + vec * stretch

        coords, traj_vector, trajectory = [[] for _ in range(3)]
        for idx, m in enumerate(markers):
            if idx == 0 and not np.all(markers[idx]["markers_head"]==0):
                M = np.stack((np.append(markers[idx]["markers_head"], 1), np.append(markers[idx]["markers_tail"], 1),
                              np.append(markers[idx]["markers_x"], 1), np.append(markers[idx]["markers_y"], 1)))
                E = np.stack((np.append(lead_positions["head_position"], 1), np.append(lead_positions["tail_position"], 1),
                              np.append(lead_positions["x_position"], 1), np.append(lead_positions["y_position"], 1)))

                X = np.linalg.lstsq(E, M, rcond=None)

                coords_mm = np.concatenate([lead_coords_mm, np.ones(shape=(lead_coords_mm.shape[0],1))], axis=1)
                coords.append((coords_mm @ X[0]).T)
                coords[idx] = coords[idx][0: 3,:]

                traj_vector.append((markers[idx]["markers_tail"] - markers[idx]["markers_head"]) / \
                      np.linalg.norm(markers[idx]["markers_tail"] - markers[idx]["markers_head"]))

                trajectory.append(np.stack((markers[idx]["markers_head"] - traj_vector[idx]*5,
                                   markers[idx]["markers_head"] + traj_vector[idx]*25)))
                trajectory[idx] = np.array((np.linspace(trajectory[idx][0, 0], trajectory[idx][1, 0], num=50),
                                            np.linspace(trajectory[idx][0, 1], trajectory[idx][1, 1], num=50),
                                            np.linspace(trajectory[idx][0, 2], trajectory[idx][1, 2], num=50))).T

        return coords, trajectory, markers, can_eldist


    def determine_rotation(self, elecModel, marker):
        """script which determines the rotation according to the markers provided"""

        rotation = elecModel["rotation"]
        normtrajvector = elecModel["normtraj_vector"]
        deg2rad = lambda x: np.pi * x / 180
        yvec = np.zeros((3,1))
        yvec[0] = -np.cos(0) * np.sin(deg2rad(rotation))
        yvec[1] = (np.cos(0) * np.cos(deg2rad(rotation))) + (np.sin(0) * np.sin(deg2rad(rotation)) * np.sin(0))
        yvec[2] = (-np.sin(0) * np.cos(deg2rad(rotation))) + (np.cos(0) * np.sin(deg2rad(rotation)) * np.sin(0))

        xvec = np.cross(yvec.T, [0,0,1])
        xvec = xvec - (np.dot(xvec, normtrajvector) / np.linalg.norm(normtrajvector)**2) * normtrajvector
        xvec = np.divide(xvec, np.linalg.norm(xvec))
        yvec = -np.cross(xvec, normtrajvector)
        yvec = -np.cross(xvec, normtrajvector)

        marker["markers_x"] = marker["markers_head"] + (xvec * elecModel["lead_diameter"]/2)
        marker["markers_y"] = marker["markers_head"] + (yvec * elecModel["lead_diameter"]/2)

        return xvec, yvec, marker, normtrajvector


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
