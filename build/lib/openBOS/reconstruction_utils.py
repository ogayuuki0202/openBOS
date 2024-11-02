import torch
import torch.nn as nn
from torch_radon import Radon
import numpy as np


class ART_torch(nn.Module):
    """
    Algebraic Reconstruction Technique (ART) model for tomography using the Radon transform.
    
    This class initializes with a sinogram and sets up the Radon transform function. 
    It includes methods for both the forward Radon transform and the backprojection.
    
    Parameters:
    sinogram (np.ndarray): The input sinogram with shape [N, Size, Angle].
    """
    
    def __init__(self, sinogram : torch.tensor):
        super(ART_torch, self).__init__()  # Call the superclass (nn.Module) initializer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sinogram = sinogram  # Input sinogram [N, Size, Angle]
        
        # Define the angles for the Radon transform
        angles = np.linspace(0, np.pi, self.sinogram.shape[1], endpoint=False)
        
        # Initialize the Radon transform function with given parameters
        self.radon_func = Radon(
            resolution=self.sinogram.shape[1], 
            angles=angles, 
            det_count=-1, 
            det_spacing=1.0, 
            clip_to_circle=False
        )
    
    # Define the Radon transform function
    def A(self, tomography: torch.tensor):
        """
        Apply the forward Radon transform to the given tomography image.
        
        Parameters:
        tomography (torch.Tensor): The input tomography image.
        
        Returns:
        torch.Tensor: The resulting sinogram after forward transformation.
        """
        return self.radon_func.forward(tomography)
    
    # Define the backprojection function
    def AT(self, sinogram : torch.tensor):
        """
        Apply the backprojection of the Radon transform to the sinogram.
        
        Parameters:
        sinogram (torch.Tensor): The input sinogram.
        
        Returns:
        torch.Tensor: The resulting tomography image after backprojection.
        """
        return self.radon_func.backprojection(sinogram)
    

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
