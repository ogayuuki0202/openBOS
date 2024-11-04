from tqdm import tqdm
import numpy as np    

def sinogram_maker_axialsymmetry(angle):
    """
    Generates a sinogram assuming axial symmetry from a single refractive angle image.
    
    Parameters:
    angle (np.ndarray): A 2D array representing the refractive angle image.
    
    Returns:
    np.ndarray: A 3D sinogram array where each slice corresponds to the refractive angle 
                projected across the height dimension, achieving axial symmetry.
    """
    # Rotate the angle image by 90 degrees
    angle = np.rot90(angle)
    height = angle.shape[1]
    
    # Initialize an empty 3D array for the sinogram
    sinogram = np.empty((angle.shape[0], height, height), dtype=angle.dtype)

    # Loop through each row in the rotated angle image
    for i, d_angle in enumerate(tqdm(angle)):
        # Broadcast each row across the height to create a symmetric 2D projection
        sinogram[i] = np.broadcast_to(d_angle[:, np.newaxis], (height, height))
        
    return sinogram
