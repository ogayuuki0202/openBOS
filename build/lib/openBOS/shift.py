from skimage.metrics import structural_similarity as ssm

def ssim(img1_array,img2_array):

    # compute the strucutural similarity matrix (SSM) on the grayscaled images
    (score, diff) = ssm(img1_array, img2_array, full=True)
    diff_inv=-diff 
    return  diff_inv