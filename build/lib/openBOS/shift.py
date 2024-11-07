from skimage.metrics import structural_similarity as ssm
import numpy as np
from PIL import Image
import openBOS.shift_utils as ib

def SSIM(ref_array : np.ndarray, exp_array : np.ndarray):
    """
    Compute the inverted Structural Similarity Index (SSIM) difference matrix between two grayscale images.

    Parameters
    ----------
    ref_array : np.ndarray
        The reference grayscale image array.
    exp_array : np.ndarray
        The experimental grayscale image array.

    Returns
    -------
    np.ndarray
        The inverted SSIM difference matrix, where higher values indicate greater dissimilarity between the two images.
    """
    # Compute the structural similarity matrix (SSM) on the grayscale images
    (score, diff) = ssm(ref_array, exp_array, full=True)
    diff_inv = -diff
    return diff_inv

def SP_BOS(ref_array : np.ndarray, exp_array : np.ndarray):
    """
    Calculate the displacement map of stripe patterns in experimental images using the Background Oriented Schlieren (BOS) method.

    Parameters
    ----------
    ref_array : np.ndarray
        The reference grayscale image array.
    exp_array : np.ndarray
        The experimental grayscale image array.

    Returns
    -------
    np.ndarray
        The displacement map with background movement compensated. Each value represents the relative movement
        of stripes between the reference and experimental images, with noise and background displacements removed.

    Notes
    -----
    The method follows these steps:
    1. Vertically stretches both reference and experimental images by a factor of 10.
    2. Binarizes the stretched images to detect stripe boundaries.
    3. Identifies upper and lower stripe boundaries and calculates stripe centers for both images.
    4. Filters noise by removing large displacement values.
    5. Computes displacement between stripe centers.
    6. Compensates for background movement by normalizing the displacement map.
    """

    im_ref=Image.fromarray(ref_array)
    im_exp=Image.fromarray(exp_array)

    #streach the image vertivally *10
    im_ref=im_ref.resize((im_ref.size[0],im_ref.size[1]*10))
    im_exp=im_exp.resize((im_exp.size[0],im_exp.size[1]*10))

    ar_ref=np.array(im_ref)
    ar_exp=np.array(im_exp)

    # Binarization
    bin_ref = ib._biner_thresh(ar_ref, 128)
    bin_exp = ib._biner_thresh(ar_exp, 128)

    print("Binarization",bin_ref.shape,bin_exp.shape)
    
    # Detect the coordinates of the color boundaries in the binarized reference image
    ref_u, ref_d = ib._bin_indexer(bin_ref)
    ref_u = np.nan_to_num(ref_u)
    ref_d = np.nan_to_num(ref_d)
    print("bin_indexer_ref",ref_u.shape,ref_d.shape)
    # Detect the coordinates of the color boundaries in the binarized experimental image
    # u represents the upper boundary of the white stripe, d represents the lower boundary
    exp_u, exp_d = ib._bin_indexer(bin_exp)
    exp_u = np.nan_to_num(exp_u)
    exp_d = np.nan_to_num(exp_d)
    print("bin_indexer_exp",exp_u.shape,exp_d.shape)

    # Remove data with abnormally large displacements as noise
    ref_u, exp_u = ib._noize_reducer_2(ref_u, exp_u, 10)
    ref_d, exp_d = ib._noize_reducer_2(ref_d, exp_d, 10)
    print("noize_reducer_2",exp_u.shape,exp_d.shape)
    print("noize_reducer_2",ref_u.shape,ref_d.shape)
    
    # Combine the upper and lower boundary data to calculate the center of the stripe
    ref = ib._mixing(ref_u, ref_d)
    exp = ib._mixing(exp_u, exp_d)

    print("mixing",ref.shape,exp.shape)
    
    # Calculate displacement (upward displacement is positive)
    diff = -(exp - ref)
    
    # Rearrange the displacement values into the correct positions and interpolate gaps
    diff_comp = ib._complementer(ref, diff)

    print("complementer",diff_comp.shape)
    
    # Subtract the overall background movement by dividing by the mean displacement
    diff_comp = diff_comp - np.nanmean(diff_comp[0:1000, 10:100])

    return diff_comp
