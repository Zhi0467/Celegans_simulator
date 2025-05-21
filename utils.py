import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import torch

def load_kato_data(n_components=1, data_dir='data/activity/AVA_HisCl'):
    """
    Load and process the Kato2015 neural activity data from traces.xlsx.
    Uses MPS (Metal Performance Shaders) if available on Apple Silicon.
    
    Parameters:
    - n_components: Number of principal components to keep (default: 1)
    - data_dir: Directory containing traces.xlsx (default: 'data/AVA_HisCl')
    
    Returns:
    - temporal_pcs: Tensor of shape (time_points, n_components) containing the temporal PCs
    - num_neurons: Number of neurons in the dataset
    """
    # Set device for Apple Silicon
    device = (torch.device("mps") 
             if torch.backends.mps.is_available() 
             else torch.device("cpu"))
    print(f"Using device: {device}")
    
    filename = 'tracesDif.xlsx'
    data_path = Path(data_dir) / filename
    
    if not data_path.exists():
        raise FileNotFoundError(f"{filename} not found in {data_dir}")
    
    print(f"Loading traces from: {data_path}")
    
    try:
        df = pd.read_excel(data_path)
        print(f"\nData from {filename}:")
        print(f"Shape: {df.shape}")
        
        # Convert to torch tensor and move to MPS if available
        traces = torch.tensor(df.values, dtype=torch.float32, device=device)
        num_neurons = traces.shape[1]
        
    except Exception as e:
        raise Exception(f"Error reading traces.xlsx: {e}")
    
    # Standardize the data on MPS
    traces_mean = traces.mean(dim=0, keepdim=True)
    traces_std = traces.std(dim=0, keepdim=True)
    traces_standardized = (traces - traces_mean) / traces_std
    
    # Transpose for neuron-space PCA
    traces_neurons = traces_standardized.T  # Shape: (neurons, timepoints)
    
    # Move to CPU for sklearn PCA
    traces_neurons_cpu = traces_neurons.cpu().numpy()
    
    # Perform PCA in neuron space
    pca = PCA(n_components=n_components)
    neuron_pcs = pca.fit_transform(traces_neurons_cpu)  # Shape: (neurons, n_components)
    
    # Move back to MPS for further computations
    neuron_pcs = torch.tensor(neuron_pcs, dtype=torch.float32, device=device)
    
    # Normalize each PC weight vector to unit length
    norms = torch.norm(neuron_pcs, dim=0, keepdim=True)  # Shape: (1, num_PCs)
    neuron_pcs_normalized = neuron_pcs / norms  # Normalize each PC (column)
    
    # Project original data onto normalized neuron PCs to get temporal PCs
    temporal_pcs = torch.matmul(traces_standardized, neuron_pcs_normalized)  # Shape: (timepoints, n_components)
    
    print(f"\nProcessing summary:")
    print(f"Original trace shape: {traces.shape}")
    print(f"Neuron PCs shape: {neuron_pcs.shape}")
    print(f"Temporal PCs shape: {temporal_pcs.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Move result back to CPU
    temporal_pcs = temporal_pcs.cpu()
    
    # Clear memory
    if device.type == 'mps':
        # Force garbage collection for MPS
        import gc
        gc.collect()
        torch.mps.empty_cache()
    
    return temporal_pcs, num_neurons
