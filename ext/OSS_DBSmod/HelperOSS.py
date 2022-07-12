#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_default_dict():
    """ loads an empty (default) version of the dictionary which is used to load data to/get information from.
    Adopted from function GUI_inp_dict in the ./OSS_platform directory  """

    dict_settings_default = dict(voxel_arr_MRI=0, voxel_arr_DTI=0, Init_neuron_model_ready=0, Init_mesh_ready=0,
                                 Adjusted_neuron_model_ready=0, CSF_mesh_ready=0, Adapted_mesh_ready=0,
                                 signal_generation_ready=0, Parallel_comp_ready=0, Parallel_comp_interrupted=0,
                                 IFFT_ready=0, MRI_data_name='icbm_avg_152_segmented.nii.gz', MRI_in_m=0,
                                 DTI_data_name='', DTI_in_m=0, CSF_index=1.0, WM_index=3.0, GM_index=2.0,
                                 default_material=3, Electrode_type='Medtronic3389', Brain_shape_name='0',
                                 x_length=40.0, y_length=40.0, z_length=40.0,
                                 Aprox_geometry_center=[10.92957028, -12.11697637, -7.69744601],
                                 Implantation_coordinate_X=10.929, Implantation_coordinate_Y=-12.117,
                                 Implantation_coordinate_Z=-7.697, Second_coordinate_X=10.929,
                                 Second_coordinate_Y=-9.437, Second_coordinate_Z=3.697, Rotation_Z=0.0,
                                 encap_thickness=0.20000000000000004, encap_tissue_type=2, encap_scaling_cond=0.8,
                                 encap_scaling_perm=0.8, pattern_model_name='0', diam_fib=[5.7], n_Ranvier=[21],
                                 v_init=-80.0, Neuron_model_array_prepared=0, Name_prepared_neuron_array='0',
                                 Global_rot=1, x_seed=10.929, y_seed=-12.117, z_seed=-7.697, x_steps=6, y_steps=0,
                                 z_steps=6, x_step=0.5, y_step=0.5, z_step=0.5,
                                 alpha_array_glob=[0.0, 0.0, 0.0, 0.0, 45, 90, 135],
                                 beta_array_glob=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 gamma_array_glob=[0.0, 45.0, 90.0, 135.0, 0.0, 0.0, 0.0], X_coord_old=0, Y_coord_old=0,
                                 Z_coord_old=0, YZ_angles=[0], ZX_angles=[0], XY_angles=[0], EQS_core='QS',
                                 Skip_mesh_refinement=0, refinement_frequency=[130.0], num_ref_freqs=-1,
                                 rel_div_CSF=5.0, CSF_frac_div=5.0, Min_Scaling=1.0, CSF_ref_reg=10.0, rel_div=5.0,
                                 Adaptive_frac_div=5.0, rel_div_current=5.0, el_order=2, number_of_processors=2,
                                 current_control=1, Phi_vector=[0.0015, None, 0.0, None], freq=130.0, T=60.0,
                                 t_step=1.0, phi=0.0, Signal_type='Rectangle', Ampl_scale=1.0, CPE_activ=0, beta=0.0,
                                 K_A=0.0, beta_ground=0.0, K_A_ground=0.0, Full_Field_IFFT=0, t_step_end=1200,
                                 VTA_from_divE=0, VTA_from_NEURON=0, VTA_from_E=0, Activation_threshold_VTA=0,
                                 spectrum_trunc_method='High Amplitude Method', trunc_param=5,
                                 Truncate_the_obtained_full_solution=0, Show_paraview_screenshots=0,
                                 Solver_Type='GMRES', FEniCS_MPI=0, Axon_Model_Type='McIntyre2002',
                                 Approximating_Dimensions=[40.0, 40.0, 40.0])

    return dict_settings_default


def rearrange_dict_settings(dict_settings):
    """ runs a series of adaptations. Adopted from function Dict_corrector in the ./OSS_platform directory  """

    # fitst, change empty lines to 0
    for key in dict_settings:
        if dict_settings[key] == '' or dict_settings[key] == '0':
            dict_settings[key] = 0

    import numpy as np
    # second, some angles will be changed to rads
    if dict_settings['Global_rot'] == 1:
        dict_settings['alpha_array_glob'] = [i * np.pi / (180.0) for i in dict_settings['alpha_array_glob']]
        dict_settings['beta_array_glob'] = [i * np.pi / (180.0) for i in dict_settings['beta_array_glob']]
        dict_settings['gamma_array_glob'] = [i * np.pi / (180.0) for i in dict_settings['gamma_array_glob']]
        dict_settings['YZ_angles'], dict_settings['XY_angles'], dict_settings['ZX_angles'] = (0, 0, 0)
    else:
        dict_settings['YZ_angles'] = [i * np.pi / (180.0) for i in dict_settings['YZ_angles']]
        dict_settings['XY_angles'] = [i * np.pi / (180.0) for i in dict_settings['XY_angles']]
        dict_settings['ZX_angles'] = [i * np.pi / (180.0) for i in dict_settings['ZX_angles']]
        dict_settings['alpha_array_glob'], dict_settings['beta_array_glob'], dict_settings['gamma_array_glob'] = (
        0, 0, 0)

    # put some variables to standard units (mm->m, ms->s and so on)
    dict_settings['T'] = dict_settings['T'] / 1000000.0
    dict_settings['t_step'] = dict_settings['t_step'] / 1000000.0
    dict_settings['phi'] = dict_settings['phi'] / 1000000.0

    # forcing to integer
    if dict_settings['spectrum_trunc_method'] == 'Octave Band Method':
        dict_settings['trunc_param'] = float(dict_settings['trunc_param'])
    else:
        dict_settings['trunc_param'] = int(dict_settings['trunc_param'])

    # one value list to int
    if isinstance(dict_settings['n_Ranvier'], list):
        if len(dict_settings['n_Ranvier']) == 1:
            dict_settings['n_Ranvier'] = int(dict_settings['n_Ranvier'][0])

    if isinstance(dict_settings['diam_fib'], list):
        if len(dict_settings['diam_fib']) == 1:
            dict_settings['diam_fib'] = float(dict_settings['diam_fib'][0])

    if isinstance(dict_settings['Aprox_geometry_center'], list):
        if len(dict_settings['Aprox_geometry_center']) == 1:
            dict_settings['Aprox_geometry_center'] = 0

    if not (isinstance(dict_settings['Approximating_Dimensions'], list)):
        dict_settings['Approximating_Dimensions'] = [0]

    if dict_settings['Neuron_model_array_prepared'] == 0:
        dict_settings['Name_prepared_neuron_array'] = 0

    # switch from percents
    dict_settings['rel_div_current'] = dict_settings['rel_div_current'] / 100.0
    dict_settings['rel_div_CSF'] = dict_settings['rel_div_CSF'] / 100.0
    dict_settings['rel_div'] = dict_settings['rel_div'] / 100.0
    dict_settings['Adaptive_frac_div'] = dict_settings['Adaptive_frac_div'] / 100
    # dict_from_GUI['CSF_frac_div']=dict_from_GUI['CSF_frac_div']/100

    return dict_settings


def build_brain_approximation(dict_settings, MRI_param):
    if dict_settings['Approximating_Dimensions'][0] == 0:  # build box or ellipsoid using MRI dimensions (starting in 0,0,0)
        x_length = abs(MRI_param.x_max - MRI_param.x_min) + MRI_param.x_vox_size
        y_length = abs(MRI_param.y_max - MRI_param.y_min) + MRI_param.y_vox_size
        z_length = abs(MRI_param.z_max - MRI_param.z_min) + MRI_param.z_vox_size
    else:  # build box or ellipsoid using given dimensions
        x_length, y_length, z_length = dict_settings['Approximating_Dimensions'][:]

    if dict_settings["Aprox_geometry_center"] == 0:  # "Centering approximation on the MRI data"
        Geom_center_x = (MRI_param.x_max + MRI_param.x_min) / 2
        Geom_center_y = (MRI_param.y_max + MRI_param.y_min) / 2
        Geom_center_z = (MRI_param.z_max + MRI_param.z_min) / 2
    else:  # centering on the given coordinates
        Geom_center_x, Geom_center_y, Geom_center_z = (dict_settings["Aprox_geometry_center"][0], dict_settings["Aprox_geometry_center"][1],
                                                       dict_settings["Aprox_geometry_center"][
                                                           2])  # this will shift only the approximating geometry, not the MRI data set!

    from Parameter_insertion import paste_geom_dim
    paste_geom_dim(x_length, y_length, z_length, Geom_center_x, Geom_center_y,
                   Geom_center_z)  # directly inserts parameters to Brain_substitute.py
    direct = os.getcwd()
    print("----- Creating brain approximation in SALOME -----")
    with open(os.devnull, 'w') as FNULL:
        subprocess.call('salome -t python3 ' + 'Brain_substitute.py' + ' --ns-port-log=' + direct + '/salomePort.txt',
                        shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    kill_SALOME_port()

    print("Brain_substitute.brep was created\n")
    with open(os.devnull, 'w') as FNULL:
        subprocess.call(
            'gmsh Meshes/Mesh_brain_substitute_max_ROI.med -3 -v 0 -o Meshes/Mesh_brain_substitute_max_ROI.msh2 && mv Meshes/Mesh_brain_substitute_max_ROI.msh2 Meshes/Mesh_brain_substitute_max_ROI.msh',
            shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    with open(os.devnull, 'w') as FNULL:
        subprocess.call(
            'dolfin-convert Meshes/Mesh_brain_substitute_max_ROI.msh Meshes/Mesh_brain_substitute_max_ROI.xml',
            shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    return x_length, y_length, z_length
