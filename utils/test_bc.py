
import os
import ants

dti_prefix = 'ep2d'

subj_folder = os.path.join("/media/storage/analysis-myoDBS/data/NIFTI/subj2/")
all_files = [filename for filename in os.listdir(subj_folder)
             if ('CT' not in filename and filename.endswith(".nii"))]

file_id_DTI = [filename for filename in os.listdir(subj_folder) if
               ('CT' not in filename and dti_prefix in filename)
               and filename.endswith('.nii')]

file_id_noDTI = [filename for filename in os.listdir(subj_folder) if
                 ('CT' not in filename and not dti_prefix in filename)
                 and filename.endswith('.nii')]
output_folder = subj_folder
# ===================================   Structural imaging  ===================================
for idx, seq in enumerate(file_id_noDTI):
    rescaler = ants.contrib.RescaleIntensity(10, 100)
    denoised_image = ants.image_read(os.path.join(subj_folder, seq))
    # denoised_image = ants.denoise_image(image=original_image, noise_model='Rician')

    min_orig, max_orig = denoised_image.min(), denoised_image.max()
    denoised_image_nonneg = rescaler.transform(denoised_image)

    bc_image_old = ants.utils.n3_bias_field_correction(denoised_image, downsample_factor=3)

    bc_image = ants.utils.n4_bias_field_correction(denoised_image,
                                                   mask=None,
                                                   shrink_factor=4,
                                                   convergence={'iters': [50, 50, 50, 50],
                                                    'tol': 1e-07},
                                                   spline_param=200,
                                                   verbose=True,
                                                   weight_mask=None)

    rescaler = ants.contrib.RescaleIntensity(min_orig, max_orig)
    bc_image = rescaler.transform(bc_image)
    save_filename = os.path.join(output_folder, subj_folder)
    ants.image_write(bc_image, filename=save_filename)
