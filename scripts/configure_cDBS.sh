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
echo "${machine}"

if [ "$machine" = Linux ]; then
packages=("xdg-utils" "ca-certificates") #TODO: are there further requisites?
  for pkg in ${packages[@]}; do

      is_pkg_installed=$(dpkg-query -W --showformat='${Status}\n' "${pkg}" | grep "install ok installed")

      if [ "${is_pkg_installed}" == "install ok installed" ]; then
          echo "${pkg}" is installed.
      fi
  done

elif [ "$machine" = Darwin ]; then
  # Check to see if Homebrew is installed, and install it if it is not
  command -v brew >/dev/null 2>&1 || { echo >&2 "Installing Homebrew Now"; \
  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"; }

  brew install --cask itk-snap
  sudo bash ./Applications/ITK-SNAP.app/Contents/bin/install_cmdl.sh # TODO: is that working at all?!

else
  echo "machines running on ${machine} are not supported"
  exit
fi

if [[ ! -d ./ext/itksnap ]]; then
    mkdir ./ext/itksnap
    wget "https://sourceforge.net/projects/itk-snap/files/itk-snap/3.6.0/itksnap-3.6.0-20170401-Linux-x86_64.tar.gz/" -O itksnap-3.6.0-20170401-Linux-x86_64.tar.gz
    mv itksnap-3.6.0-20170401-Linux-x86_64.tar.gz ./ext/itksnap/
fi

# ==============================    create general architecture of folders    ==============================

(if [[ ! -d ./data ]]; then
    mkdir ./data
fi

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

if [[ ! -d ./templates/mni_icbm152_nlin_asym_09b ]]; then
    mkdir ./templates/mni_icbm152_nlin_asym_09b
    wget "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip" -O temp.zip
    unzip -o temp.zip -d /templates/mni_icbm152_nlin_asym_09b/
    rm -rf temp.zip
fi
)

# ==============================    create subfolders in data folder    ==============================
(
cd data

# create the relevant folders for data storage
if [[ ! -d ./DICOM ]]; then
    mkdir ./DICOM
fi

if [[ ! -d ./NIFTI ]]; then
    mkdir ./NIFTI
    touch ./NIFTI/subjdetails.csv
fi
)

# copy blank version of the configuration file
cp ./private/config_imagingTBdef.yaml config_imagingTB.yaml

# TODO: At the end of the installation, a window should appear with further instrcutions: 1. install antspy/antspynet,
# TODO: 2. install ITK-snap and run itk-snap 3.8 command line wrapper once, 3. copy templates but espcially icbm152 to
# TODO: corresponding folder

