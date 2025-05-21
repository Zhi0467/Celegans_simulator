# load and preprocess the connectome and activity datasets 

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import torch
import pandas as pd
import itertools
import os
import plotly.graph_objects as go
import plotly.io as pio
import math # For ceil
import matplotlib.pyplot as plt

def gaussian_smooth(data, sigma=1.0):
    """
    Apply Gaussian smoothing to time series data along the time dimension.
    
    Parameters:
    - data: torch.Tensor of shape [n_neurons, n_timepoints]
    - sigma: Standard deviation of the Gaussian kernel
    
    Returns:
    - Smoothed data tensor of the same shape
    """
    # Ensure data is on CPU for numpy operations
    device = data.device
    data_np = data.cpu().numpy()
    
    # Create Gaussian kernel
    # Window size is 6*sigma (3 sigma on each side) to capture ~99.7% of the Gaussian
    window_size = int(6 * sigma)
    if window_size % 2 == 0:  # Make window size odd
        window_size += 1
    
    # If window_size is too large for the data, reduce it
    if window_size > data_np.shape[1]:
        window_size = min(data_np.shape[1], 5)
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            return data  # Too small for smoothing
    
    # Create the Gaussian kernel
    half_win = window_size // 2
    x = np.arange(-half_win, half_win + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution to each neuron's time series
    smoothed_data = np.zeros_like(data_np)
    for i in range(data_np.shape[0]):
        # Pad the time series for proper convolution
        padded = np.pad(data_np[i], (half_win, half_win), mode='edge')
        # Apply convolution
        smoothed_data[i] = np.convolve(padded, kernel, mode='valid')
    
    # Convert back to torch tensor and return
    return torch.tensor(smoothed_data, dtype=torch.float32, device=device)

def load_white_data(data_dir='data/connectome/White1986', 
                    filter_motor_neurons=True, 
                    motor_neuron_prefixes=('DB', 'VB', 'DA', 'VA', 'DD', 'VD')):
    """
    Load the C. elegans connectome data.
    
    Parameters:
    - data_dir: Directory containing connectome data files
    - filter_motor_neurons: Whether to filter out motor neurons (default: True)
    - motor_neuron_prefixes: Prefixes identifying motor neurons to exclude
    
    Returns:
    - matrices: Dictionary containing connectivity matrices for each connection type
    - neuron_index: Dictionary mapping neuron names to indices
    """
    # Load the neuron names from the connectome_EJ.csv file
    ej_path = Path(data_dir) / 'connectome_EJ.csv'
    ej_df = pd.read_csv(ej_path, index_col=0)
    
    # Extract neuron names from the index of the EJ dataframe
    neurons = list(ej_df.index)
    
    # Filter out motor neurons if requested
    if filter_motor_neurons and motor_neuron_prefixes:
        original_count = len(neurons)
        filtered_neurons = []
        
        for neuron in neurons:
            # Check if the neuron name starts with any of the motor neuron prefixes
            if not any(neuron.startswith(prefix) for prefix in motor_neuron_prefixes):
                filtered_neurons.append(neuron)
        
        neurons = filtered_neurons
        
    # Create mapping from neuron names to indices
    n = len(neurons)
    neuron_index = {neuron: i for i, neuron in enumerate(neurons)}
    
    # Load the .xls file to get the connection data
    file_path = Path(data_dir) / 'Celeganconnect.xls'
    df = pd.read_excel(file_path)
    
    # Filter out rows where either Neuron 1 or Neuron 2 is not in our neuron list
    df = df[df['Neuron 1'].isin(neurons) & df['Neuron 2'].isin(neurons)]
    
    # Initialize matrices for each connection type and a combined matrix
    matrix_types = df['Type'].unique()
    matrices = {conn_type: np.zeros((n, n)) for conn_type in matrix_types}
    combined_matrix = np.zeros((n, n))
    
    # Populate the matrices
    for _, row in df.iterrows():
        if row['Neuron 1'] in neuron_index and row['Neuron 2'] in neuron_index:
            i = neuron_index[row['Neuron 1']]
            j = neuron_index[row['Neuron 2']]
            conn_type = row['Type']
            weight = row['Nbr']
            
            if conn_type in ['EJ', 'Rp']:  # Bidirectional synapses
                matrices[conn_type][i, j] += weight
                matrices[conn_type][j, i] += weight
                combined_matrix[i, j] += weight
                combined_matrix[j, i] += weight
            elif conn_type in ['Sp', 'R', 'S']:  # Directed synapses
                matrices[conn_type][i, j] += weight
                combined_matrix[i, j] += weight
            elif conn_type == 'NMJ':  # Neuromuscular junctions
                matrices[conn_type][i, j] += weight
                combined_matrix[i, j] += weight
    
    # Create DataFrames from the matrices
    matrix_dfs = {conn_type: pd.DataFrame(mat, index=neurons, columns=neurons)
                 for conn_type, mat in matrices.items()}
    combined_df = pd.DataFrame(combined_matrix, index=neurons, columns=neurons)
    
    # Save matrices to CSV files
    for conn_type, df in matrix_dfs.items():
        path = Path(data_dir) / f"connectome_{conn_type}.csv"
        df.to_csv(path)
    
    combined_df.to_csv(Path(data_dir) / "connectome_combined.csv")
    
    return matrices, neuron_index

def load_receptor_synaptic_data(csv_path = 'data/connectome/GeneCon.xlsx',
                            white_data_dir='data/connectome/White1986',
                            filter_motor_neurons=True,
                            motor_neuron_prefixes=('DB', 'VB', 'DA', 'VA', 'DD', 'VD')):
    """
    Loads chemical synaptic connectivity data from Cook et al. (via the provided CSV),
    applies sign based on the 'Sign' column, filters out 'complex'/'no pred' signs,
    and formats it into a matrix aligned with the neuron order from load_white_data.

    Parameters:
    - csv_path: Path to the 'Table-Export.xls - Table Export.csv' file.
    - white_data_dir: Directory for White1986 connectome data (for reference frame).
    - filter_motor_neurons: Whether to filter out motor neurons (must match reference).
    - motor_neuron_prefixes: Prefixes identifying motor neurons (must match).

    Returns:
    - synaptic_matrix: NxN numpy array representing directed signed chemical synapse weights,
                       aligned with the neuron order from load_white_data.
                       Values are 0 if sign was 'complex' or 'no pred'.
    - neuron_index: Dictionary mapping neuron names to indices from load_white_data.
    """

    # 1. Get the reference neuron list and ordering
    try:
        _, neuron_index = load_white_data(
            data_dir=white_data_dir,
            filter_motor_neurons=filter_motor_neurons,
            motor_neuron_prefixes=motor_neuron_prefixes
        )
    except FileNotFoundError as e:
        print(f"Error: White1986 data not found in {white_data_dir}.")
        raise e
    except Exception as e:
        print(f"Error loading reference data: {e}")
        raise e

    neurons = list(neuron_index.keys())
    n = len(neurons)

    # 2. Load the synaptic data from the CSV
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Synaptic data CSV not found at: {csv_path}")

    try:
        df_synaptic = pd.read_excel(csv_file)
        sender_col, receiver_col, weight_col, type_col, sign_col = 'Source', 'Target', 'Edge Weight', 'Edge Type', 'Sign'
        required_cols = [sender_col, receiver_col, weight_col, type_col, sign_col]
        if not all(col in df_synaptic.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Need: {required_cols}. Found: {df_synaptic.columns.tolist()}")
        # Convert sign column to lowercase for consistent comparison
        df_synaptic[sign_col] = df_synaptic[sign_col].str.lower()
    except Exception as e:
        raise Exception(f"Error reading/processing CSV file {csv_path}: {e}")

    # 3. Filter the synaptic data
    # Keep only chemical synapses
    df_chemical = df_synaptic[df_synaptic[type_col].str.lower() == 'chemical'].copy()

    # --- MODIFIED FILTERING ---
    # Keep only connections where both neurons are in our reference list
    # AND where the sign is explicitly '+' or '-'
    valid_signs = ['+', '-']
    df_filtered = df_chemical[
        df_chemical[sender_col].isin(neuron_index) &
        df_chemical[receiver_col].isin(neuron_index) &
        df_chemical[sign_col].isin(valid_signs) # Only keep rows with '+' or '-' sign
    ]

    # 4. Create the connectivity matrix
    synaptic_matrix = np.zeros((n, n), dtype=float)

    # Populate the matrix
    for _, row in df_filtered.iterrows():
        sender = row[sender_col]
        receiver = row[receiver_col]
        weight = row[weight_col]
        sign = row[sign_col] # Sign is already lowercased

        # Get indices from the reference neuron_index
        i = neuron_index[sender]
        j = neuron_index[receiver]

        # --- MODIFIED WEIGHT ASSIGNMENT ---
        signed_weight = float(weight)
        if sign == '-':
            signed_weight *= -1
        # Add the signed weight
        synaptic_matrix[i, j] += signed_weight
        # --- END MODIFIED WEIGHT ASSIGNMENT ---

    # 5. Return the matrix and the consistent neuron index
    return synaptic_matrix, neuron_index

def load_synaptic_data(csv_file_path = 'data/connectome/GeneCon.xlsx', 
                       white_dir = 'data/connectome/White1986', 
                       filter_motor_neuron = True,
                       target_norm = 15.0):
        try:
            # --- Load Matrix A (Signed Synaptic from Cook et al.) ---
            # Uses the modified load_cook_synaptic_data that incorporates sign
            matrix_A, neuron_index = load_receptor_synaptic_data(
                csv_path=csv_file_path,
                white_data_dir=white_dir,
                filter_motor_neurons=filter_motor_neuron# Ensure consistency
            )


            # --- Load Matrix B (e.g., EJ or S from White et al.) ---
            white_matrices, white_neuron_index = load_white_data(
                data_dir=white_dir,
                filter_motor_neurons=filter_motor_neuron
            )
            assert neuron_index == white_neuron_index # Verify alignment

            # Select the desired matrix B from White data
            matrix_B_key = 'S' # Example: Choose 'S' for synaptic connections
                            # Or use 'EJ' for gap junctions, or 'combined_matrix'
            # Find the combined matrix from load_white_data return if needed:
            # _, _, _, _, combined_matrix_B = load_white_data(...) # If using the internal combined one

            # Get the matrix corresponding to matrix_B_key
            matrix_B = white_matrices.get(matrix_B_key)

            if matrix_B is None:
                print(f"Warning: Matrix type '{matrix_B_key}' not found in White data. Cannot perform fill-in. Using only Matrix A.")
                matrix_C = matrix_A.copy() # Use only A if B is not found
            else:
                # Ensure B is a numpy array aligned with neuron_index
                if isinstance(matrix_B, pd.DataFrame):
                    matrix_B = matrix_B.loc[list(neuron_index.keys()), list(neuron_index.keys())].values
                matrix_B = np.asarray(matrix_B)

                valid_mask = (matrix_A != 0) & (matrix_B != 0)

                # 2. start with a zero matrix
                matrix_C = np.zeros_like(matrix_A)

                # 3. wherever valid, copy in A (or whatever combination you like)
                matrix_C[valid_mask] = matrix_B[valid_mask]

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

        return white_matrices, neuron_index

# Function to visualize the matrices
def plot_matrix_comparison_heatmap(csv_file_path = 'data/connectome/GeneCon.xlsx', 
                                   white_dir = 'data/connectome/White1986', 
                                   white_matrix_key='Unknown White Matrix',
                                   filter_motor_neuron = True,
                                   save_path=None):
    """
    Generates side-by-side heatmaps for comparing three synaptic matrices.

    Args:
        matrix_white (np.ndarray): Matrix loaded from load_white_data (e.g., 'S' or 'EJ').
        matrix_cook_signed (np.ndarray): Signed synaptic matrix from load_receptor_synaptic_data.
        matrix_combined (np.ndarray): Matrix resulting from combining the other two.
        neuron_index (dict): Dictionary mapping neuron names to indices.
        white_matrix_key (str): Name of the white matrix used (e.g., 'S', 'EJ') for title.
        save_path (str or Path, optional): Path to save the figure. If None, displays the plot.
    """
    matrix_cook_signed, neuron_index = load_receptor_synaptic_data(
        csv_path=csv_file_path,
        white_data_dir=white_dir,
        filter_motor_neurons=filter_motor_neuron# Ensure consistency
    )
    matrix_white, neuron_index = load_white_data(data_dir=white_dir, filter_motor_neurons=filter_motor_neuron)
    matrix_white = matrix_white['S']
    matrix_combined, _ = load_synaptic_data(csv_file_path=csv_file_path, white_dir=white_dir, filter_motor_neuron=filter_motor_neuron)
    neuron_names = list(neuron_index.keys())
    n = len(neuron_names)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 1 row, 3 columns, adjusted size slightly

    # --- Determine Colormap and Value Limits based on Combined Matrix ---
    cmap = 'coolwarm' # Diverging colormap for all plots

    # Find the max absolute value in the COMBINED matrix to set symmetric limits
    max_abs_val_combined = np.max(np.abs(matrix_combined))
    if max_abs_val_combined == 0:
        max_abs_val_combined = 1.0 # Avoid zero range if combined matrix is all zeros
        print("Warning: Combined matrix has zero range. Using default scale [-1, 1].")

    vmin_shared = -max_abs_val_combined
    vmax_shared = max_abs_val_combined
    print(f"Using shared diverging scale: [{vmin_shared:.2f}, {vmax_shared:.2f}] based on combined matrix.")

    # --- Plot Heatmaps with Shared Scale ---
    common_kwargs = {
        'cmap': cmap,
        'vmin': vmin_shared,
        'vmax': vmax_shared,
        'aspect': 'auto',
        'interpolation': 'nearest' # Often better for matrices than default
    }

    # 1. White Matrix (using the shared scale)
    im0 = axes[0].imshow(matrix_white, **common_kwargs)
    axes[0].set_title(f"White et al. ('{white_matrix_key}')")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Weight/Count')

    # 2. Cook Signed Matrix (using the shared scale)
    im1 = axes[1].imshow(matrix_cook_signed, **common_kwargs)
    axes[1].set_title("Cook et al. Signed Synaptic (A)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Signed Weight')

    # 3. Combined Matrix (using the shared scale)
    im2 = axes[2].imshow(matrix_combined, **common_kwargs)
    axes[2].set_title(f"Combined (A, filled by B='{white_matrix_key}')")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Combined Signed Weight')

    # --- Formatting ---
    tick_step = max(1, n // 10) # Show ~10 ticks maximum
    for ax in axes:
        ax.set_xlabel("Target Neuron Index")
        ax.set_yticks(np.arange(0, n, tick_step)) # Set y-ticks for the first plot only
        if ax == axes[0]:
             ax.set_ylabel("Source Neuron Index")
        else:
             ax.set_yticklabels([]) # Hide y-labels for subsequent plots
        ax.set_xticks(np.arange(0, n, tick_step))
        ax.tick_params(axis='x', rotation=90) # Rotate x-ticks if needed

    fig.suptitle("Synaptic Matrix Comparison (Shared Scale)", fontsize=16)
    # Use constrained_layout for better spacing, especially with colorbars
    # plt.constrained_layout(rect=[0, 0.03, 1, 0.95])

    # --- Save or Show ---
    if save_path:
        # Ensure save_path is a Path object if needed
        if isinstance(save_path, str): save_path = Path(save_path)
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Matrix comparison heatmap saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

# preprocess missing neural activity data 
# so that all target neurons have timeseries data for later use
"""
functionally similar neurons can be used to replace each other, they are identified at:
- https://pnicompneurojc.github.io/papers/Linderman%202019.pdf
- https://github.com/mmtree/Celegans_premotor/blob/main/functions/functions_regression.ipynb
"""

def data_replacement(i, nm, names_have, switchout_list, dict_nms, ts_myneuros, dtsdt_myneuros, Fts1, dFts1dt):
    """
    Replaces missing neuron data with data from bilateral counterparts or functionally related neurons.

    Parameters:
        i (int): Index of the neuron in the dataset.
        nm (str): Name of the neuron (e.g., "AVAL", "RIBL").
        names_have (list): List of neuron names present in the dataset.
        switchout_list (list): List of neuron names eligible for replacement.
        dict_nms (dict): Dictionary mapping neuron names to their indices in the dataset.
        ts_myneuros (numpy.ndarray): Matrix containing time-series data for neurons.
        dtsdt_myneuros (numpy.ndarray): Matrix containing time derivatives of the time-series data.
        Fts1 (numpy.ndarray): Reference dataset containing time-series data for all neurons.
        dFts1dt (numpy.ndarray): Reference dataset containing time derivatives of the time-series data.
    """
    # If the neuron is directly available, use its data
    if nm in names_have:
        ts_myneuros[i, :] = Fts1[dict_nms[nm]]
        dtsdt_myneuros[i, :] = dFts1dt[dict_nms[nm]]
        return

    # Otherwise, look for replacements
    if len(nm) == 4 and nm in switchout_list:
        #print(f"switchout {nm}")
        nm_new = ""
        if nm[-1] == "L":
            nm_new = nm[:3] + "R"
        elif nm[-1] == "R":
            nm_new = nm[:3] + "L"

        if nm_new in names_have:  # Twin neuron exists in the dataset
            #print(nm_new)
            ts_myneuros[i, :] = Fts1[dict_nms[nm_new]]
            dtsdt_myneuros[i, :] = dFts1dt[dict_nms[nm_new]]

        elif nm == "AVBL":  # If AVBL is missing and has no twin, substitute with RIB
            print("AVBL isn't here and doesn't have twin")
            if "RIBL" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIBL"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIBL"]]
            elif "RIBR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIBR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIBR"]]

        elif nm == "AVBR":
            print("AVBR isn't here and doesn't have twin")
            if "RIBR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIBR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIBR"]]
            elif "RIBL" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIBL"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIBL"]]

        elif nm == "RIBL":  # If RIBL is missing and has no twin, substitute with AVB
            print("RIBL isn't here and doesn't have twin")
            if "AVBL" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["AVBL"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBL"]]
            elif "AVBR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["AVBR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBR"]]

        elif nm == "RIBR":
            print("RIBR isn't here and doesn't have twin")
            if "AVBR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["AVBR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBR"]]
            elif "AVBL" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["AVBL"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBL"]]

        elif nm == "AIBL":  # If AIBL is missing, substitute with RIM
            print("AIBL isn't here and doesn't have twin")
            if "RIML" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIML"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIML"]]
            elif "RIMR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIMR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIMR"]]

        elif nm == "AIBR":
            print("AIBR isn't here and doesn't have twin")
            if "RIMR" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIMR"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIMR"]]
            elif "RIML" in names_have:
                ts_myneuros[i, :] = Fts1[dict_nms["RIML"]]
                dtsdt_myneuros[i, :] = dFts1dt[dict_nms["RIML"]]

    elif len(nm) == 5 and nm in switchout_list:
        #print(f"switchout {nm}")
        nm_new = ""
        if nm[-1] == "L":
            nm_new = nm[:4] + "R"
        elif nm[-1] == "R":
            nm_new = nm[:4] + "L"

        if nm_new in names_have:  # Twin neuron exists in the dataset
            #print(nm_new)
            ts_myneuros[i, :] = Fts1[dict_nms[nm_new]]
            dtsdt_myneuros[i, :] = dFts1dt[dict_nms[nm_new]]

    elif nm == "RID":
        if "AVBR" in names_have:
            ts_myneuros[i, :] = Fts1[dict_nms["AVBR"]]
            dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBR"]]
        elif "AVBL" in names_have:
            ts_myneuros[i, :] = Fts1[dict_nms["AVBL"]]
            dtsdt_myneuros[i, :] = dFts1dt[dict_nms["AVBL"]]

def load_kato_data(data_dir='data/activity/WT_NoStim', sheet_name=0, smooth=True, sigma=15.0):
    """
    Load and process the Kato2015 neural activity data.
    
    Parameters:
    - data_dir: Directory containing IDs.xlsx, traces.xlsx, and tracesDif.xlsx
    - sheet_name: Which sheet to read from Excel files (default: 0)
    - smooth: Whether to apply Gaussian smoothing (default: True)
    - sigma: Standard deviation of the Gaussian kernel for smoothing (default: 1.0)
    """
    switchout_list = [
        "AVAL", "AVAR", "AVBL", "AVBR", 
        "RIBL", "RIBR", "AIBL", "AIBR",
        "RIML", "RIMR", "RID"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                                  ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Load connectome data to get neuron_index
    connectome_matrices, neuron_index = load_white_data()  # Get neuron_index from connectome

    # Load IDs with specified sheet
    ids_path = Path(data_dir) / 'IDs.xlsx'
    if not ids_path.exists():
        raise FileNotFoundError(f"IDs.xlsx not found in {data_dir}")
    
    try:
        ids_df = pd.read_excel(ids_path, header=None, sheet_name=sheet_name)
        
        # Filter to only include proper neuron names (exclude numbers and invalid entries)
        neuron_ids = {}
        for idx, name in enumerate(ids_df.iloc[0]):
            if pd.notna(name):
                name_str = str(name).strip("'").strip()
                # Only include if it's not empty, not just brackets, and not a number
                if (name_str and name_str != '[]' and 
                    not name_str.isdigit() and  # Exclude pure numbers
                    any(c.isalpha() for c in name_str)):  # Must contain at least one letter
                    neuron_ids[name_str] = idx
        
    except Exception as e:
        raise Exception(f"Error reading IDs.xlsx: {e}")
    
    # Load traces with specified sheet
    traces_path = Path(data_dir) / 'traces.xlsx'
    traces_dif_path = Path(data_dir) / 'tracesDif.xlsx'

    if not traces_path.exists():
        raise FileNotFoundError(f"traces.xlsx not found in {data_dir}")
    
    try:
        traces_df = pd.read_excel(traces_path, sheet_name=sheet_name)

        traces = torch.tensor(traces_df.values, dtype=torch.float32, device=device)
        traces_dif_df = pd.read_excel(traces_dif_path, sheet_name=sheet_name)
        traces_dif = torch.tensor(traces_dif_df.values, dtype=torch.float32, device=device)
        
        # Standardize both datasets
        traces_mean = traces.mean(dim=0, keepdim=True)
        traces_std = traces.std(dim=0, keepdim=True)
        traces_standardized = (traces - traces_mean) / traces_std

        traces_dif_mean = traces_dif.mean(dim=0, keepdim=True)
        traces_dif_std = traces_dif.std(dim=0, keepdim=True)
        traces_dif_standardized = (traces_dif - traces_dif_mean) / traces_dif_std
        
        # Move to CPU and transpose to (neurons, timepoints)
        traces_np = traces_standardized.cpu().numpy().T
        traces_dif_np = traces_dif_standardized.cpu().numpy().T
        
        traces = torch.tensor(traces_np, dtype=torch.float32, device=device)
        traces_dif = torch.tensor(traces_dif_np, dtype=torch.float32, device=device)
        
        # Apply Gaussian smoothing if requested
        if smooth:
            traces = gaussian_smooth(traces, sigma)
            # No need to smooth derivatives if we're recalculating them

        # Initialize arrays for core neurons
        ts_myneuros = np.zeros((len(switchout_list), traces_np.shape[1]))
        dtsdt_myneuros = np.zeros((len(switchout_list), traces_dif_np.shape[1]))
        
        # Apply data replacement for core neurons
        names_have = list(neuron_ids.keys())
        for i, name in enumerate(switchout_list):
            data_replacement(
                i, name, names_have, switchout_list, neuron_ids,
                ts_myneuros, dtsdt_myneuros,
                traces_np, traces_dif_np
            )
        
        # Check for zero rows after data replacement
        zero_rows = np.where(np.all(ts_myneuros == 0, axis=1))[0]
        if len(zero_rows) > 0:
            missing_neurons = [switchout_list[i] for i in zero_rows]
            #raise ValueError(f"Missing activity data for core neurons: {missing_neurons}. "
            #                f"These neurons have no data and no suitable replacements.")
        
        # Convert back to torch tensors
        core_traces = torch.tensor(ts_myneuros, dtype=torch.float32, device=device)
        core_traces_dif = torch.tensor(dtsdt_myneuros, dtype=torch.float32, device=device)

        # Create a new tensor for permuted traces based on connectome IDs
        permuted_traces = torch.zeros((len(neuron_index), core_traces.shape[1]), device=device)
        for name, idx in neuron_ids.items():
            if name in neuron_index:
                connectome_id = neuron_index[name]
                permuted_traces[connectome_id] = traces[idx]
                neuron_ids[name] = connectome_id

    except Exception as e:
        raise Exception(f"Error processing traces: {e}")
    
    # Load fps with specified sheet
    file_path = Path(data_dir) / 'fps.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"fps.xlsx not found in {data_dir}")
    
    try:
        fps_df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
        fps = round(float(fps_df.iloc[0, 0]), 4)
    except Exception as e:
        raise Exception(f"Error reading fps value: {e}")

    # Clear memory
    if device.type == 'mps':
        import gc
        gc.collect()
        torch.mps.empty_cache()
    
    return permuted_traces, core_traces_dif, neuron_ids, switchout_list, fps

def load_full_neural_data(data_dir='data/activity/WT_NoStim', sheet_name=0, smooth=True, sigma=15.0):
    """
    Load full neural activity data with optional Gaussian smoothing.
    
    Parameters:
    - data_dir: Directory containing IDs.xlsx, traces.xlsx, and tracesDif.xlsx
    - sheet_name: Which sheet to read from Excel files
    - smooth: Whether to apply Gaussian smoothing (default: True) 
    - sigma: Standard deviation of the Gaussian kernel for smoothing (default: 1.0)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 
                                  ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load connectome data to get neuron_index
    connectome_matrices, neuron_index = load_white_data()  # Get neuron_index from connectome

    # Load IDs with specified sheet
    ids_path = Path(data_dir) / 'IDs.xlsx'
    if not ids_path.exists():
        raise FileNotFoundError(f"IDs.xlsx not found in {data_dir}")
    
    try:
        ids_df = pd.read_excel(ids_path, header=None, sheet_name=sheet_name)
        
        # Filter to only include proper neuron names (exclude numbers and invalid entries)
        neuron_ids = {}
        for idx, name in enumerate(ids_df.iloc[0]):
            if pd.notna(name):
                name_str = str(name).strip("'").strip()
                # Only include if it's not empty, not just brackets, and not a number
                if (name_str and name_str != '[]' and 
                    not name_str.isdigit() and  # Exclude pure numbers
                    any(c.isalpha() for c in name_str)):  # Must contain at least one letter
                    neuron_ids[name_str] = idx

    except Exception as e:
        raise Exception(f"Error reading IDs.xlsx: {e}")

    # Load traces with specified sheet
    traces_path = Path(data_dir) / 'traces.xlsx'
    traces_dif_path = Path(data_dir) / 'tracesDif.xlsx'

    if not traces_path.exists():
        raise FileNotFoundError(f"traces.xlsx not found in {data_dir}")
    
    try:
        traces_df = pd.read_excel(traces_path, sheet_name=sheet_name)

        traces = torch.tensor(traces_df.values, dtype=torch.float32, device=device)
        traces_dif_df = pd.read_excel(traces_dif_path, sheet_name=sheet_name)
        traces_dif = torch.tensor(traces_dif_df.values, dtype=torch.float32, device=device)

        # Standardize both datasets
        traces_mean = traces.mean(dim=0, keepdim=True)
        traces_std = traces.std(dim=0, keepdim=True)
        traces_standardized = (traces - traces_mean) / traces_std

        traces_dif_mean = traces_dif.mean(dim=0, keepdim=True)
        traces_dif_std = traces_dif.std(dim=0, keepdim=True)
        traces_dif_standardized = (traces_dif - traces_dif_mean) / traces_dif_std
        
        # Move to CPU and transpose to (neurons, timepoints)
        traces_np = traces_standardized.cpu().numpy().T
        traces_dif_np = traces_dif_standardized.cpu().numpy().T

        traces = torch.tensor(traces_np, dtype=torch.float32, device=device)
        traces_dif = torch.tensor(traces_dif_np, dtype=torch.float32, device=device)
        
        # Apply Gaussian smoothing if requested
        if smooth:
            traces = gaussian_smooth(traces, sigma)
            # No need to smooth derivatives if we're recalculating them
            
            # If needed, we could recalculate derivatives here based on smoothed traces
            # instead of using the provided derivatives
        
        # Create a new tensor for permuted traces based on connectome IDs
        permuted_traces = torch.zeros((len(neuron_index), traces_np.shape[1]), device=device)
        for name, idx in neuron_ids.items():
            if name in neuron_index:
                connectome_id = neuron_index[name]
                permuted_traces[connectome_id] = traces[idx]
                neuron_ids[name] = connectome_id

    except Exception as e:
        raise Exception(f"Error processing traces: {e}")

    # Load fps with specified sheet
    file_path = Path(data_dir) / 'fps.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"fps.xlsx not found in {data_dir}")
    
    try:
        fps_df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
        fps = round(float(fps_df.iloc[0, 0]), 4)
    except Exception as e:
        raise Exception(f"Error reading fps value: {e}")
    # Clear memory
    if device.type == 'mps':
        import gc
        gc.collect()
        torch.mps.empty_cache()
    return permuted_traces, traces_dif, neuron_ids, fps
