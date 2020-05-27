





class extract_electrode_location:
    "This file extracts electrodes in neuroradiological imaging to present differentiated their location in the brain."

#affirmation whether necessary files are avilable as niiCT

#CT-standart range check (e.g. -1024, 4096)

#identifaction of total metal components in imaging (Metal_Threshold)

#determine brain mask -> available or not ? If yes access to parameters

#detection of metal artifacts inside images, start with maskedimage as complete image without process

#removal of the skull -> accentuate of brainmMask just to show the brain without bony structure

#merge the largest connected components of metal inside the brain

#specificate  number of connected components and confirmation of connection

#access to pixel list and sorting them (Pixel values? Size?)

#[... step will follow]

#guessing the number and idxs of electrodes in image

#determine the mean of voxelnumber

#definition of axis length

#if nElecs=0 error display

#create output structure and try associate electrodes from xml defitions

#with pixelList presentation of Electrode location

