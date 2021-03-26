#!/usr/bin/env bash

# ==============================    Check for OS and install required packages   ==============================
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo ${machine}

if [ $machine=Linux ]; then
packages=("xdg-utils" "ca-certificates") #TODO: are there further requisites
  for pkg in ${packages[@]}; do

      is_pkg_installed=$(dpkg-query -W --showformat='${Status}\n' ${pkg} | grep "install ok installed")

      if [ "${is_pkg_installed}" == "install ok installed" ]; then
          echo ${pkg} is installed.
      fi
  done
  apt-get -qq update && apt-get -qq --yes --force-yes install itksnap # TODO: maybe not working b/c of lacking root priv.

elif [ $machine=Darwin ]; then
  # Check to see if Homebrew is installed, and install it if it is not
  command -v brew >/dev/null 2>&1 || { echo >&2 "Installing Homebrew Now"; \
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"; }

  brew install --cask itk-snap
  sudo bash ./Applications/ITK-SNAP.app/Contents/bin/install_cmdl.sh # TODO: is that working at all?!

else
  echo "machines running on ${machine} are not supported"
  exit
fi

# TODO: should ITK-snap be included here?

# ==============================    create general architecture of folders    ==============================
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

# ==============================    create subfolders in data folder    ==============================
cd data

# create the relevant folders for data storage
if [[ ! -d ./DICOM ]]; then
    mkdir ./DICOM
fi

if [[ ! -d ./NIFTI ]]; then
    mkdir ./NIFTI
    touch ./NIFTI/subjdetails.csv
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

