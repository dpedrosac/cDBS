import utils.HelperFunctions as HF
# GUItoolbox


def runDCM2NII():
    text = 'Separate GUI to transform DICOM files using Chris Rordens routines ' \
           'see http://www.nitrc.org/projects/dcm2nii/'
    return HF.LittleHelpers.split_lines(text)


def displayRAWdata():
    text = 'Display the data available for the subj selected, to see what is present. Note: if more than one ' \
           'subject is selected an error will be dropped!'
    return HF.LittleHelpers.split_lines(text)


def renameFolders():
    text = 'Displays a small GUI which enables to change the prefix for all folders. This new prefix is saved in ' \
           'the configuration file'
    return HF.LittleHelpers.split_lines(text)


def subjectDetails():
    text = 'Displays the subject details and the names of the subjects corresponding to the prefix'
    return HF.LittleHelpers.split_lines(text)


# GUIdcm2nii
def saveDirButton():
    text = "The DICOM folder is saved into a configuration file, which is read when GUI is loaded in order to " \
           "facilitate continuing with previuos session (file located in main folder of Toolbox)"
    return HF.LittleHelpers.split_lines(text)


# SettingsDCM2NII
def includeFilesDCM2NII():
    text = "Here, a list of sequences can be defined which are kept after DICOM files are transformed. The idea " \
           "is that all 'localizers' and sequences of no interest are deleted, to mantain the list of files " \
           "uncluttered; However, by leaving it blank all files are kept, e.g. to see which sequences are available"
    return HF.LittleHelpers.split_lines(text)


def CompressionDCM2NII():
    text = "This option sets whether compression is active or. For the time being the amount of compression cannot " \
           "be changed"
    return HF.LittleHelpers.split_lines(text)


def LabelFilenameDCM2NII():
    text = "Please enter here the prefix/file information that should be included when saving NIFTI files. " \
           "Characters such as '<', '>', ':', '/' ... MUST be avoided (default is %p_%s). " \
           "From documentation:\n\t %a : antenna (coil) number inserted. For example, the output filename" \
           " 'myName%a' would generate 'myName1', 'myName2', each for each coil. Note that most scans combine data " \
           "from all coils and in these cases this option is ignored. For example, most scans which combine " \
           "data from all coils would simply be called 'myName'\n\t%d :" \
           "series description (0008,103E) inserted. For example, an echo-planar image " \
           "converted with 'myName%d' would yield 'myNameEPI' \n\t%e : echo number " \
           "inserted. For example, a sequence with two echo times converted with the output" \
           "filename 'myName%e' will yield 'myName1' and 'myName2'. Note that most MRI" \
           " sequences only use a single echo time, and in these cases you would only get " \
           "'myName1'.\n\t%f : input folder name inserted. For example, the output " \
           "filename'myName%f' combined with an input folder '/usr/Subj22' will result in " \
           "the output file named'myNameSubj22.nii' \n\t%i : patient ID " \
           "(DICOM tag 0010,0020) inserted. For example, the output filename 'myName%i' " \
           " would convert an image where the patient ID is named 'ID123' to be " \
           " 'myNameID123.nii' \n\t%m : manufacturer name For example, the output filename" \
           " 'myName%m' would convert an image from a GE scanner to 'myNameGE.nii', while " \
           " an image from  Philips would be 'myNamePh.nii', whereas Siemens would be " \
           " 'myNameSi.nii', otherwise the manufacturer is not available ('myNameNA.nii')." \
           " (requires dcm2nii versions from 2015 or later). \n\t%n : subject name (DICOM" \
           " tag 0010,0010) inserted. For example, the output filename 'myName%n' would " \
           " convert an image from John Doe to 'myNameJohnDoe.nii'. This option works best" \
           " if your participant names use only English letters, for other European " \
           " languages you may find it makes some basic conversions ('Müller' will become " \
           " 'Muller'). For non-European languages you will find this option unsatisfactory" \
           ". Perhaps future versions can support DICOM tag 0008,0005. \n\t %p: protocol" \
           " name (DICOM tag 0018,1030) inserted. For example, the output filename" \
           " 'myName%p' would convert image where protocol is named T1 to be 'myNameT1.nii'" \
           " \n\t%q: sequence name (DICOM tag 0018,1020) inserted.For example," \
           " the output filename 'myName%q' would convert a Spin Echo sequence to be " \
           "'myNameSE.nii' (new feature, in versions from 30Aug2015).\n\t%s : series (DICOM " \
           "tag 0020,0011) inserted. For example, the output filename 'myName%s' would " \
           "convert the second series to be 'myName2.nii'. If you want to zero-pad the " \
           "series number, insert the number of digits desired (0..9). For example applying " \
           "the filter 'm%s' when converting 11 series will create files that will cause" \
           " problems for a simple alphabetical sort, e.g. 'm1.nii,m11.nii,m2.nii...m9.nii'." \
           " In contrast specifying 'm%3s' will help sorting (e.g. 'm001.nii,m002.nii" \
           "...m011.nii').\n\t%t : session date and time inserted (DICOM tags 0008,0021 and " \
           "0008,0030). For example, the output filename 'myName%t' would convert an image " \
           "where the session began at 1:23pm on 13 Jan 2014 as 'myName20140113132322.nii' " \
           "\n\t%z : Sequence Name (0018,0024) inserted, so a T1 scan converted with " \
           "'myName%z' might yield 'myNameT1'."

    return HF.LittleHelpers.split_lines(text)


def LabelBIDS():
    text = "This option sets whether sidecar is stored or not (By default, data is always anonimised and BIDS " \
           "are always stored). "

    return HF.LittleHelpers.split_lines(text)


def LabelVerbosity():
    text = "According to the documentation this defines whether dcm2nii should be silent (0), with numerous" \
           " output (1) or logorrhoeic (2); default = 1"

    return HF.LittleHelpers.split_lines(text)


def LabelReorientCrop():
    text = "This option sets whether reorientation according to header and cropping should be performed or not"

    return HF.LittleHelpers.split_lines(text)


# SettingsNIFTIprocAnts
def LabelPrefixBias():
    text = "The prefix for the file resulting from bias correction. This may be necessary for debugging purposes"

    return HF.LittleHelpers.split_lines(text)


def LabelShrink():
    text = "To lessen computation time, input images may be resampled. Shrink factor, specified as single " \
           "integers, describe this resampling (factors <= 4 commonly used)."

    return HF.LittleHelpers.split_lines(text)
