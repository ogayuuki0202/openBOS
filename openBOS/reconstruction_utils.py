import torch
import torch.nn as nn
from torch_radon import Radon
import numpy as np

class ART(nn.Module):
    """
    Algebraic Reconstruction Technique (ART) model for tomography using the Radon transform.
    
    This class initializes with a sinogram and sets up the Radon transform function. 
    It includes methods for both the forward Radon transform and the backprojection.
    
    Parameters:
    sinogram (np.ndarray): The input sinogram with shape [N, Size, Angle].
    """
    
    def __init__(self, sinogram : torch.tensor):
        super(ART, self).__init__()  # Call the superclass (nn.Module) initializer
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