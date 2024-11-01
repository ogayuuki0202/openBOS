from tqdm.contrib import tenumerate
from tqdm import tqdm
import numpy as np
import torch

def SOR_2D(tensor_laplacian: torch.tensor, batch_size: int,device:str, omega_SOR: float,e: float,tolerance:float =1e-24,max_stable_iters:int=1000000):
    """
    Perform Successive Over-Relaxation (SOR) on a Laplacian array using GPU.

    This function applies SOR to solve linear systems iteratively for each batch of a 
    Laplacian input array. If a GPU is available, computations are performed on the GPU for efficiency.

    Parameters:
    tensor_laplacian (torch.tensor): Input Laplacian array with shape [N, Ly, Lz].
    batch_size (int): Number of samples per batch.If you use CPU for the processing,Batchsize=1 is recomennded.
    device (str) : 'cuda' or 'cpu'
    omega_SOR (float): Relaxation factor for SOR, controls the convergence speed.
    e (float): Tolerance for stopping the iterative process based on residual.
    tolerance (float): The difference threshold for loss change to consider convergence stable.
    max_stable_iters (int): Maximum number of iterations with stable residuals allowed for convergence.

    Returns:
    np.ndarray: The SOR-processed array concatenated across all batches.
    """
    # Convert the input array to a torch tensor and move it to the selected device, set up data loader
    tensor_laplacian_dataloader = torch.utils.data.DataLoader(
        torch.tensor(tensor_laplacian).to(device), batch_size=batch_size, shuffle=False
    )

    # Extract dimensions from the input Laplacian array
    Ly = tensor_laplacian.shape[1]
    Lz = tensor_laplacian.shape[2]

   
    # Initialize variables
    u_list = []  # List to store results for each batch

    # Iterate over each batch in the data loader
    for batch_idx, batch in enumerate(tensor_laplacian_dataloader):
        # Move the batch to the device
        slice_laplacian = batch.to(device)
        batch_size, Ly, Lz = slice_laplacian.size()
        
        # Initialize u for SOR iterations
        u = torch.zeros([batch_size, Ly, Lz], device=device)
        delta = 1.0
        stable_count = 0  # Reset stable count for each batch
        prev_delta = float('inf')  # Initialize previous delta

        # SOR Iterative Loop
        while delta > e and stable_count < max_stable_iters:
            # Save current state for convergence check
            u_in = u.clone()
            
            # Perform SOR update on the inner region
            u[:, 1:-1, 1:-1] = u[:, 1:-1, 1:-1] + omega_SOR * (
                (u_in[:, 2:, 1:-1] + u_in[:, :-2, 1:-1] + u_in[:, 1:-1, 2:] + u_in[:, 1:-1, :-2] 
                 + slice_laplacian[:, 1:-1, 1:-1]) / 4 - u[:, 1:-1, 1:-1]
            )

            # Set boundary conditions
            u[:, 0, :] = 0
            u[:, Ly-1, :] = 0
            u[:, :, 0] = 0
            u[:, :, Lz-1] = 0

            # Compute max absolute change (delta) for convergence
            delta = torch.max(torch.abs(u - u_in))
            
            # Check if residual change is within tolerance to count as stable
            if abs(delta - prev_delta) < tolerance:
                stable_count += 1
            else:
                stable_count = 0

            prev_delta = delta  # Update previous delta for next iteration

            # Print iteration information
            print("\r", f'Iteration: {n_iter}, Loss: {delta} Stable Count: {stable_count}', end="")

            # Update iteration count
            n_iter += 1

        # Append result for the batch to the list
        u_list.append(u)

    # Concatenate results for all batches and return as a single tensor
    u_tensor = torch.cat(u_list, dim=0)

    return np.array(u_tensor)