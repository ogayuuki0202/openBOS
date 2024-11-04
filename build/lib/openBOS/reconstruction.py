import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip      
from skimage.transform import radon, iradon                        

def abel_transform(angle: np.ndarray, center: float, ref_x: float, G: float):
    """
    Perform the Abel transform to convert angle values into density differences.

    This function applies the Abel transform to a given array of angle values,
    compensating for background movement, calculating distances from the center,
    and integrating to obtain density differences using the Gladstone-Dale constant.

    Parameters:
    ----------
    angle : np.ndarray
        The input array of angle values for transformation.
    center : float
        The center position for the transformation, defining the region of interest.
    ref_x : float
        The x-coordinate used to offset the background movement.
    G : float
        The Gladstone-Dale constant used to convert the result to density differences.

    Returns:
    -------
    np.ndarray
        The resulting array of density differences obtained from the Abel transform.
    """
    
    # Offset the angle values by subtracting the mean value at the reference x-coordinate
    angle = angle - np.mean(angle[0:round(angle.shape[0]/20), ref_x])
    
    # Remove values below the center since they are not used in the calculation
    angle = angle[0:center]
    
    # Reverse the angle array so that the upper end becomes the central axis
    angle = angle[::-1]

    # Calculate the distance from the central axis (η)
    eta = np.array(range(angle.shape[0]))
    
    # Initialize an array to store the results
    ans = np.zeros_like(angle)

    # Calculate the values outward from r=0
    for r in tqdm(range(center)):
        # A: Denominator √(η² - r²)
        # Calculate η² - r²
        A = eta**2 - r**2
        # Trim the array to keep the integration range (we extend to r+1 to avoid division by zero)
        A = A[r+1:center]
        # Take the square root to obtain √(η² - r²)
        A = np.sqrt(A)
        # Reshape for broadcasting
        A = np.array([A]).T
        
        # B: The integrand (1/π * ε/√(η² - r²))
        B = angle[r+1:center] / (A * np.pi)
        # Sum B vertically to perform integration
        ans[r] = B.sum(axis=0)
    
    # Convert the result (difference in refractive index Δn) to density difference Δρ
    density = ans / G

    return density

def ART(sinogram, mu, e, bpos=True):
    """
    Perform Algebraic Reconstruction Technique (ART) to reconstruct images from a sinogram.

    Parameters:
    - sinogram (ndarray): The input sinogram, where each row corresponds to a projection at a specific angle.
    - mu (float): The relaxation parameter that controls the update step size in the reconstruction process.
    - e (float): The convergence threshold for the maximum absolute error in the reconstruction.
    - bpos (bool): If True, enforces non-negative constraints on the pixel values in the reconstruction.

    Returns:
    - x_list (list of ndarray): A list of reconstructed images, one for each projection set in the sinogram.
    """
    
    N = 1  # Initial grid size for reconstruction
    ANG = 180  # Total rotation angle for projections
    VIEW = sinogram[0].shape[0]  # Number of views (angles) in each projection
    THETA = np.linspace(0, ANG, VIEW + 1)[:-1]  # Angles for radon transform
    pbar = tqdm(total=sinogram[0].shape[0], desc="Initialization", unit="task")

    # Find the optimal N that matches the projection dimensions
    while True:
        x = np.ones((N, N))  # Initialize a reconstruction image with ones

        def A(x):
            # Forward projection (Radon transform)
            return radon(x, THETA, circle=False).astype(np.float32)

        def AT(y):
            # Backprojection (inverse Radon transform)
            return iradon(y, THETA, circle=False, output_size=N).astype(np.float32) / (np.pi/2 * len(THETA))
        
        ATA = AT(A(np.ones_like(x)))  # ATA matrix for scaling

        # Check if the current grid size N produces projections of the correct shape
        if A(x).shape[0] == sinogram[0].shape[0]:
            break

        # Adjust N in larger steps if the difference is significant, else by 1
        if sinogram[0].shape[0] - A(x).shape[0] > 20:
            N += 10
        else:
            N += 1

        # Update progress bar
        pbar.n = A(x).shape[0]
        pbar.refresh()
    pbar.close()

    loss = np.inf
    x_list = []

    # Process each projection set in the sinogram
    for i in tqdm(range(sinogram.shape[0]), desc='Process', leave=True):
        b = sinogram[i]  # Current projection data
        ATA = AT(A(np.ones_like(x)))  # Recalculate ATA for current x
        loss = float('inf')  # Reset loss

        # Iteratively update x until convergence
        while np.max(np.abs(loss)) > e:
            # Compute the update based on the difference between projection and reconstruction
            loss = np.divide(AT(b - A(x)), ATA)
            x = x + mu * loss

        x_list.append(x)  # Append the reconstructed image for the current projection

    return x_list
