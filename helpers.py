import pandas as pd
import numpy as np
from numpy.linalg import svd, matrix_rank
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr

# Function to compute the Frobenius norm
def frobenius_norm(original, masked):
    return np.linalg.norm(original - masked)

# Function to compute Cosine Similarity
def cosine_similarity(original, masked):
    # Flatten the matrices into vectors
    return 1 - cosine(original.flatten(), masked.flatten())

# Function to compute Structural Similarity Index (SSIM)
def structural_similarity_index(original, masked):
    data_range = original.max() - original.min()  # Calculate the range of values
    return ssim(original, masked, data_range=data_range)

# Compute Rank Correlation (Spearman's rank correlation)
def rank_correlation(original, masked):
    return spearmanr(original.flatten(), masked.flatten())[0]

def normalized_effective_rank(matrix):
    """
    Calculate the normalized effective rank of a matrix by normalizing by the full rank.
    """
    u, s, vh = svd(matrix, full_matrices=False)  # Singular Value Decomposition
    s = s[s > 1e-10]  # Remove singular values close to zero to avoid numerical issues
    normalized_s = s / np.sum(s)  # Normalize singular values
    entropy = -np.sum(normalized_s * np.log(normalized_s))  # Shannon entropy
    effective_rank = np.exp(entropy)  # Exponential of the entropy
    
    full_rank = matrix_rank(matrix)  # Full rank of the matrix (number of non-zero singular values)
    
    # Normalize by the full rank
    return effective_rank / full_rank if full_rank > 0 else 0  # Avoid division by zero

def wasserstein_distance_normalized(A, B):
    eigenvalues_A = np.linalg.eigvals(A)
    eigenvalues_B = np.linalg.eigvals(B)
    distance = np.sum(np.abs(np.sort(eigenvalues_A) - np.sort(eigenvalues_B)))
    # Normalize by the maximum eigenvalue magnitude
    normalization_factor = np.sum(np.abs(np.abs(eigenvalues_A)) + np.abs(np.abs(eigenvalues_B)))
    return distance / normalization_factor

def trace_norm_distance_normalized(A, B):
    singular_values_A = np.linalg.svd(A, compute_uv=False)
    singular_values_B = np.linalg.svd(B, compute_uv=False)
    distance = np.sum(np.abs(np.sort(singular_values_A) - np.sort(singular_values_B)))
    # Normalize by the sum of singular values of A and B
    normalization_factor = np.sum(np.abs(singular_values_A)) + np.sum(np.abs(singular_values_B))
    return distance / normalization_factor

def participation_ratio(matrix):
    """
    Computes the participation ratio of the singular values of a matrix.
    
    Parameters:
        matrix (numpy.ndarray): The input matrix.
    
    Returns:
        float: The participation ratio of the singular values.
    """
    # Compute singular values of the matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    
    # Compute S^2 and S^4
    squared_singular_values = singular_values ** 2
    fourth_power_singular_values = singular_values ** 4
    
    # Averages ⟨S^2⟩ and ⟨S^4⟩
    mean_squared = np.mean(squared_singular_values)
    mean_fourth = np.mean(fourth_power_singular_values)
    
    # Compute participation ratio
    participation_ratio = (mean_squared ** 2) / mean_fourth

    print(f"The first 5 singular values are {singular_values[:5]}")
    
    if mean_fourth == 0:
        return 0  # Handle edge case where all singular values are zero
    
    return participation_ratio