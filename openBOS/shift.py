from skimage.metrics import structural_similarity as ssm
import numpy as np
from PIL import Image
import openBOS.shift_utils as ib

def SSIM(ref_array : np.ndarray, exp_array : np.ndarray):
    """
    Compute the inverted structural similarity matrix (SSM) between two grayscale images.

    Parameters:
    ----------
    ref_array : np.ndarray
        The reference grayscale image array.
    exp_array : np.ndarray
        The experimental grayscale image array.

    Returns:
    -------
    np.ndarray
        The inverted difference matrix showing dissimilarity between the two images.
    """
    # Compute the structural similarity matrix (SSM) on the grayscale images
    (score, diff) = ssm(ref_array, exp_array, full=True)
    diff_inv = -diff
    return diff_inv

def SP_BOS(ref_array : np.ndarray, exp_array : np.ndarray):
    """
    Calculate the displacement of stripes in experimental images using the BOS (Background Oriented Schlieren) method.

    Parameters:
    ----------
    ref_array : np.ndarray
        The reference grayscale image array.
    exp_array : np.ndarray
        The experimental grayscale image array.

    Returns:
    -------
    np.ndarray
        The compensated displacement map showing the relative movement of stripes, with the overall background movement removed.

    Notes:
    -----
    The function processes the reference and experimental images by:
    - Binarizing the images
    - Detecting stripe boundaries
    - Removing noise from large displacements
    - Calculating displacement between stripe centers in the reference and experimental images
    - Compensating for overall background movement
    """

    im_ref=Image.fromarray(ref_array)
    im_exp=Image.fromarray(exp_array)

    #streach the image vertivally *10
    im_ref=im_ref.resize((im_ref.size[0],im_ref.size[1]*10))
    im_exp=im_exp.resize((im_exp.size[0],im_exp.size[1]*10))

    print("resize",im_ref.shape,im_exp.shape)

    ar_ref=np.array(im_ref)
    ar_exp=np.array(im_exp)

    # Binarization
    bin_ref = ib.biner_thresh(ar_ref, 128)
    bin_exp = ib.biner_thresh(ar_exp, 128)

    print("Binarization",bin_ref.shape,bin_exp.shape)
    
    # Detect the coordinates of the color boundaries in the binarized reference image
    ref_u, ref_d = ib.bin_indexer(bin_ref)
    ref_u = np.nan_to_num(ref_u)
    ref_d = np.nan_to_num(ref_d)
    print("bin_indexer_ref",ref_u.shape,ref_d.shape)
    # Detect the coordinates of the color boundaries in the binarized experimental image
    # u represents the upper boundary of the white stripe, d represents the lower boundary
    exp_u, exp_d = ib.bin_indexer(bin_exp)
    exp_u = np.nan_to_num(exp_u)
    exp_d = np.nan_to_num(exp_d)
    print("bin_indexer_exp",exp_u.shape,exp_d.shape)

    # Remove data with abnormally large displacements as noise
    ref_u, exp_u = ib.noize_reducer_2(ref_u, exp_u, 10)
    ref_d, exp_d = ib.noize_reducer_2(ref_d, exp_d, 10)
    print("noize_reducer_2",exp_u.shape,exp_d.shape)
    print("noize_reducer_2",ref_u.shape,ref_d.shape)
    
    # Combine the upper and lower boundary data to calculate the center of the stripe
    ref = ib.mixing(ref_u, ref_d)
    exp = ib.mixing(exp_u, exp_d)

    print("mixing",ref.shape,exp.shape)
    
    # Calculate displacement (upward displacement is positive)
    diff = -(exp - ref)
    
    # Rearrange the displacement values into the correct positions and interpolate gaps
    diff_comp = ib.complementer(ref, diff)

    print("complementer",diff_comp.shape)
    
    # Subtract the overall background movement by dividing by the mean displacement
    diff_comp = diff_comp - np.nanmean(diff_comp[0:1000, 10:100])

    return diff_comp
