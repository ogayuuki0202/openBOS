import numpy as np
import openBOS.shift_utils as ib
from metpy.units import units
from metpy.calc import density
from tqdm import tqdm,trange

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

def get_G(temperature, pressure, humidity):
    """
    Calculate the Gladstone constant based on temperature, pressure, and humidity.

    Parameters:
    temperature (float): The temperature in degrees Celsius (°C).
    pressure (float): The pressure in hectopascals (hPa).
    humidity (float): The humidity as a percentage (%).

    Returns:
    float: The calculated Gladstone-Dale constant (G).
    """

    # Calculate the density using the given pressure, temperature, and humidity
    density_inf = density(pressure * units.hPa, temperature * units.degC, humidity * units.percent)

    n_inf = 1.0003  # Refractive index of air
    G = (n_inf - 1) / density_inf  # Gladstone-Dale Relation
    return G

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

def compute_laplacian_chunk_2D(array_chunk:np.ndarray):
    """
    Computes the Laplacian of a given chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (ndarray): A chunk of the input array to compute the Laplacian on.

    Returns:
    ndarray: The Laplacian computed for the given chunk.
    """
    grad_yy = np.gradient(array_chunk, axis=1)  # Compute gradient along the y-axis
    grad_zz = np.gradient(array_chunk, axis=2)  # Compute gradient along the z-axis
    laplacian_chunk = grad_yy + grad_zz         # Sum gradients to approximate Laplacian
    return laplacian_chunk

def compute_laplacian_chunk_3D(array_chunk:np.ndarray):
    """
    Computes the Laplacian of a given chunk by calculating gradients 
    along specific axes and summing them.
    
    Parameters:
    array_chunk (ndarray): A chunk of the input array to compute the Laplacian on.

    Returns:
    ndarray: The Laplacian computed for the given chunk.
    """
    grad_xx = np.gradient(array_chunk, axis=0)
    grad_yy = np.gradient(array_chunk, axis=1)  # Compute gradient along the y-axis
    grad_zz = np.gradient(array_chunk, axis=2)  # Compute gradient along the z-axis
    laplacian_chunk = grad_xx+ grad_yy + grad_zz         # Sum gradients to approximate Laplacian
    return laplacian_chunk

def compute_laplacian_in_chunks_2D(array:np.ndarray, chunk_size:int = 100):
    """
    Computes the Laplacian of an input array in smaller chunks, allowing for 
    memory-efficient processing of large arrays.
    
    Parameters:
    array (ndarray): Input array on which to compute the Laplacian.
    chunk_size (int): Size of each chunk to split the array into for processing.

    Returns:
    ndarray: The Laplacian of the input array, computed in chunks.
    """
    # Get the shape of the input array
    shape = array.shape
    
    # Initialize an array to store the Laplacian result
    laplacian = np.zeros_like(array)
    
    # Process the array in chunks
    for i in trange(0, shape[0], chunk_size):          # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):       # Loop over y-axis in chunks
            for k in range(0, shape[2], chunk_size):   # Loop over z-axis in chunks
                # Extract the current chunk
                chunk = array[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size]
                
                # Compute the Laplacian for the current chunk
                laplacian_chunk = compute_laplacian_chunk_2D(chunk)
                
                # Place the result in the corresponding location of the main array
                laplacian[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size] = laplacian_chunk
    
    return laplacian

def compute_laplacian_in_chunks_3D(array:np.ndarray, chunk_size:int = 100):
    """
    Computes the Laplacian of an input array in smaller chunks, allowing for 
    memory-efficient processing of large arrays.
    
    Parameters:
    array (ndarray): Input array on which to compute the Laplacian.
    chunk_size (int): Size of each chunk to split the array into for processing.

    Returns:
    ndarray: The Laplacian of the input array, computed in chunks.
    """
    # Get the shape of the input array
    shape = array.shape
    
    # Initialize an array to store the Laplacian result
    laplacian = np.zeros_like(array)
    
    # Process the array in chunks
    for i in trange(0, shape[0], chunk_size):          # Loop over x-axis in chunks
        for j in range(0, shape[1], chunk_size):       # Loop over y-axis in chunks
            for k in range(0, shape[2], chunk_size):   # Loop over z-axis in chunks
                # Extract the current chunk
                chunk = array[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size]
                
                # Compute the Laplacian for the current chunk
                laplacian_chunk = compute_laplacian_chunk_3D(chunk)
                
                # Place the result in the corresponding location of the main array
                laplacian[i:i+chunk_size, j:j+chunk_size, k:k+chunk_size] = laplacian_chunk
    
    return laplacian