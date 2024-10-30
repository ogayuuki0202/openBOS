import numpy as np
from tqdm import tqdm

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
    angle = angle - np.mean(angle[0:200, ref_x])

    print(center)
    
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
