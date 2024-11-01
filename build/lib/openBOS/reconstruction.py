import numpy as np
from tqdm import tqdm
import torch      
from .reconstruction_utils import ART_torch  
from tqdm.contrib import tzip                              

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
    angle = angle - np.mean(angle[0:angle.shape[0]/20, ref_x])
    
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

def ART(sinogram: np.ndarray, batch_size: int, device:str, eps: float,tolerance:float =1e-24,max_stable_iters:int=1000000):
    """
    Perform Algebraic Reconstruction Technique (ART) on a sinogram using GPU.
    
    This function applies ART for tomographic reconstruction by iteratively adjusting 
    predictions to reduce the residual between predictions and target sinogram values. 
    The process runs on GPU if available for faster computation.
    
    Parameters:
    sinogram (np.ndarray): Input sinogram with shape [N, Size, Angle].
    batch_size (int): Number of samples per batch.If you use CPU for the processing,Batchsize=1 is recomennded.
    device (str) : 'cuda' or 'cpu'
    eps (float): Tolerance for stopping the iterative process based on residual.

    tolerance (float): The difference threshold for loss change to consider convergence stable.
    max_stable_iters (int): Maximum number of iterations with stable residuals allowed for convergence.

    Returns:
    torch.Tensor: Reconstructed image tensor concatenated across all processed batches.
    """


    # Convert sinogram to a torch tensor and move it to the selected device
    sinogram_tensor = torch.FloatTensor(sinogram).permute(0, 2, 1).to(device)

    # Create data loaders for target and initial predictions
    target_dataloader = torch.utils.data.DataLoader(sinogram_tensor, batch_size=batch_size, shuffle=False)
    predict_dataloader = torch.utils.data.DataLoader(torch.zeros_like(sinogram_tensor), batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"target": target_dataloader, "predict": predict_dataloader}

    # Initialize the ART model with the input sinogram
    model = ART_torch(sinogram=sinogram)

    # Extract data loaders
    predict_dataloader = dataloaders_dict["predict"]
    target_dataloader = dataloaders_dict["target"]

    processed_batches = []

    # Convergence parameters

    prev_loss = float('inf')

    # Iterate through the data loader batches
    for i, (predict_batch, target_batch) in enumerate(tzip(predict_dataloader, target_dataloader)):
        # Move batches to the device
        predict_batch = predict_batch.to(model.device)
        target_batch = target_batch.to(model.device)
        stable_count = 0  # Counter for stable iterations

        iter_count = 0
        ATA = model.AT(model.A(torch.ones_like(predict_batch)))  # Precompute ATA for normalization
        ave_loss = torch.inf  # Initialize average loss

        # Initial loss calculation
        loss = torch.divide(model.AT(target_batch - model.A(predict_batch)), ATA)
        ave_loss = torch.max(torch.abs(loss)).item()

        # ART Iterative Reconstruction Loop
        while ave_loss > eps and stable_count < max_stable_iters:
            predict_batch = predict_batch + loss  # Update prediction
            ave_loss = torch.max(torch.abs(loss)).item()
            print("\r", f'Iteration: {iter_count}, Residual: {ave_loss}, Stable Count: {stable_count}', end="")
            iter_count += 1

            # Recalculate loss
            loss = torch.divide(model.AT(target_batch - model.A(predict_batch)), ATA)

            # Check residual change to update stable count
            if abs(ave_loss - prev_loss) < tolerance:
                stable_count += 1
            else:
                stable_count = 0

            prev_loss = ave_loss

        processed_batches.append(predict_batch)

    # Concatenate all processed batches along the batch dimension and return
    return torch.cat(processed_batches, dim=0)
