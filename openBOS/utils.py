import numpy as np
import openBOS.shift_utils as ib
from metpy.units import units
from metpy.calc import density
from tqdm import tqdm,trange
import torch

def shift2angle(shift: np.ndarray, ref_array: np.ndarray, sensor_pitch: float, resolution_of_pattern: float, Lb: float, Lci: float):
    """
    Convert the background image displacement to the angle of light refraction.

    Parameters:
    shift (np.ndarray): Displacement values from the background image.
    ref_array (np.ndarray): Reference image array used for calculations.
    sensor_pitch (float): The pitch of the image sensor in meters.
    resolution_of_pattern (float): The resolution of the pattern in meters per pixel.
    Lb (float): Distance from the background to the object being captured.
    Lci (float): Distance from the image sensor to the object being captured.

    Returns:
    tuple: 
        - angle (np.ndarray): The calculated angles of light refraction.
        - Lc (float): The distance from the object to the lens.
        - Li (float): The distance from the lens to the image sensor.
        - projection_ratio (float): The ratio of projection based on the dimensions.
    """
    
    # Size of one LP (in pixels)
    dpLP = ib.cycle(ref_array)

    sensor_pitch = sensor_pitch * 10**-3  # Convert sensor pitch from mm to m
    BGmpLP = 1 / resolution_of_pattern * 10**-3  # Convert pattern resolution from mm to m

    # Size of one LP on the projection plane (m/LP)
    mpLP = dpLP * sensor_pitch

    # Magnification of the imaging
    projection_ratio = mpLP / BGmpLP

    # Total length
    Lbi = Lci + Lb

    Lc = Lbi / (projection_ratio + 1) - Lb  # Distance from the object to the lens
    Li = Lci - Lc  # Distance from the lens to the image sensor

    # Calculate the angle based on shift and projection properties
    angle = shift * (sensor_pitch) / (projection_ratio * Lb)
    np.nan_to_num(angle, copy=False)  # Replace NaN values with zero in the angle array

    return angle, Lc, Li, projection_ratio

def get_gladstone_dale_constant(temperature, pressure, humidity):
    """
    Calculate the Gladstone-Dale constant based on temperature, pressure, and humidity.

    Parameters:
    temperature (float): Temperature in degrees Celsius (°C).
    pressure (float): Pressure in hectopascals (hPa).
    humidity (float): Humidity as a percentage (%).

    Returns:
    tuple: (G, density) where
        - G (float): The calculated Gladstone-Dale constant.
        - density (float): The density of the atmasphere.

    """

    # Calculate the gas density based on the provided pressure, temperature, and humidity
    density_air = density(pressure * units.hPa, temperature * units.degC, humidity * units.percent)

    n_air = 1.0003  # Refractive index of air
    G = (n_air - 1) / density_air  # Calculate the Gladstone-Dale constant based on the relation

    return G,density_air


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

def compute_laplacian_chunk_2D(array_chunk: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian of a given 2D chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (Tensor): A chunk of the input tensor to compute the Laplacian on.

    Returns:
    Tensor: The Laplacian computed for the given chunk.
    """
    grad_yy = array_chunk[:, 2:] - 2 * array_chunk[:, 1:-1] + array_chunk[:, :-2]  # y-axis
    grad_zz = array_chunk[2:, :] - 2 * array_chunk[1:-1, :] + array_chunk[:-2, :]  # z-axis
    laplacian_chunk = grad_yy + grad_zz
    return laplacian_chunk

def compute_laplacian_chunk_3D(array_chunk: torch.Tensor) -> torch.Tensor:
    """
    Computes the Laplacian of a given 3D chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (Tensor): A chunk of the input tensor to compute the Laplacian on.

    Returns:
    Tensor: The Laplacian computed for the given chunk.
    """
    grad_xx = array_chunk[2:, :, :] - 2 * array_chunk[1:-1, :, :] + array_chunk[:-2, :, :]
    grad_yy = array_chunk[:, 2:, :] - 2 * array_chunk[:, 1:-1, :] + array_chunk[:, :-2, :]
    grad_zz = array_chunk[:, :, 2:] - 2 * array_chunk[:, :, 1:-1] + array_chunk[:, :, :-2]
    laplacian_chunk = grad_xx + grad_yy + grad_zz
    return laplacian_chunk

def compute_laplacian_in_chunks_2D(array: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
    """
    Computes the Laplacian of an input 2D tensor in smaller chunks, allowing for 
    memory-efficient processing of large tensors.
    """
    shape = array.shape
    laplacian = torch.zeros_like(array)

    for i in trange(0, shape[0], chunk_size):            # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):         # Loop over y-axis in chunks
            chunk = array[i:i + chunk_size, j:j + chunk_size]
            laplacian_chunk = compute_laplacian_chunk_2D(chunk)
            laplacian[i+1:i + chunk_size-1, j+1:j + chunk_size-1] = laplacian_chunk

    return laplacian

def compute_laplacian_in_chunks_3D(array: torch.Tensor, chunk_size: int = 100) -> torch.Tensor:
    """
    Computes the Laplacian of an input 3D tensor in smaller chunks, allowing for 
    memory-efficient processing of large tensors.
    """
    shape = array.shape
    laplacian = torch.zeros_like(array)

    for i in trange(0, shape[0], chunk_size):            # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):         # Loop over y-axis in chunks
            for k in range(0, shape[2], chunk_size):     # Loop over z-axis in chunks
                chunk = array[i:i + chunk_size, j:j + chunk_size, k:k + chunk_size]
                laplacian_chunk = compute_laplacian_chunk_3D(chunk)
                laplacian[i+1:i + chunk_size-1, j+1:j + chunk_size-1, k+1:k + chunk_size-1] = laplacian_chunk

    return laplacian