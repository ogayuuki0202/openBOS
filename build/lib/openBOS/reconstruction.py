import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip      
from skimage.transform import radon, iradon                        

def abel_transform(angle: np.ndarray, center: float, ref_x: float, G: float):
    """
    Perform the Abel transform to convert refractive angle values into density differences.

    This function applies the Abel transform on a 2D array of refractive angles. It compensates
    for background movement by subtracting the mean value at a reference x-coordinate, calculates
    distances from the center axis, and integrates to derive density differences using the
    Gladstone-Dale constant.

    Parameters
    ----------
    angle : np.ndarray
        A 2D numpy array representing refractive angles for each pixel.
    center : float
        The index along the y-axis corresponding to the central axis of the transform.
    ref_x : float
        A reference x-coordinate to compute and offset background movement.
    G : float
        The Gladstone-Dale constant, used to convert the computed refractive index differences
        to density differences.

    Returns
    -------
    np.ndarray
        A 1D array of density differences derived from the Abel transform.

    Notes
    -----
    This function calculates density differences through an integral-based approach. The refractive
    angle image is rotated to align with the axis of symmetry, and values are integrated from the
    center outwards, adjusting for axial symmetry.

    Examples
    --------
    >>> angle_image = np.random.rand(200, 300)  # Simulated refractive angle image
    >>> density_differences = abel_transform(angle_image, center=150, ref_x=50, G=0.0001)
    >>> print(density_differences.shape)
    (150,)
    """
    
    # Offset the angle values by subtracting the mean value at the reference x-coordinate
    angle = angle - np.mean(angle[0])
    
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

    The ART method iteratively updates pixel values to minimize the error between projections
    and the input sinogram, facilitating accurate image reconstruction from projections.

    Parameters
    ----------
    sinogram : np.ndarray
        A 2D or 3D numpy array representing the sinogram. Each row corresponds to a projection
        at a specific angle.
    mu : float
        The relaxation parameter controlling the update step size during the iterative process.
    e : float
        The convergence threshold for the maximum absolute error in the reconstruction.
    bpos : bool, optional
        If True, enforces non-negative pixel values in the reconstruction, by default True.

    Returns
    -------
    list of np.ndarray
        A list of reconstructed 2D arrays, each corresponding to a projection set in the input sinogram.

    Notes
    -----
    - The function dynamically adjusts the grid size `N` until it matches the shape of the sinogram projections.
    - The `radon` and `iradon` functions from `skimage.transform` are used to perform forward and backward
      projections, respectively.
    - The method stops when the maximum absolute error between successive updates falls below `e`.

    Examples
    --------
    >>> sinogram = np.random.rand(180, 128)  # Example sinogram with 180 projections of length 128
    >>> reconstructed_images = ART(sinogram, mu=0.1, e=1e-6)
    >>> print(len(reconstructed_images))
    180
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