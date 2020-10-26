#!/bin/bash

# ---------------------------------------------
cd data

# create the relevant folders for data storage
if [[ ! -d ./DICOM ]]; then
    mkdir ./DICOM
fi

if [[ ! -d ./NIFTI ]]; then
    mkdir ./NIFTI
fi

# creates a folder to save (general) log files
cd ..
if [[ ! -d ./logs ]]; then
    mkdir ./logs
fi

if [[ ! -d ./ext ]]; then
    mkdir ./ext
fi

cd ext
if [[ ! -d ./templates ]]; then
    mkdir ./templates
fi

cd ..

# copy blank version of the configuration file
cp ./private/config_imagingTBdef.yaml config_imagingTB.yaml

# cd temp
# wget "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_minc2.zip" -O temp.zip
# unzip temp.zip
# rm temp.zip

# TODO: At the end of the installation, a window should appear with further instrcutions: 1. install antspy/antspynet,
# TODO: 2. install ITK-snap and run itk-snap 3.8 command line wrapper once, 3. copy templates but espcially icbm152 to
# TODO: corresponding folder

