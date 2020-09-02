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

# copy blank version of the configuration file
cp ./private/config_imagingTBdef.yaml config_imagingTB.yaml

# TODO: run itk-snap 3.8 command line wrapper once

