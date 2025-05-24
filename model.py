'''
This file contains the IdentifiedPCA class, which is a subclass of torch.utils.data.Dataset.
It is used to load the identified neural activity data, smooth it, 
relabel neurons to match connectome and perform PCA on it.

The TrialPCA class uses LinearDynamics as the simulator with
a trainable linear projection to latent space, and it contains
methods for 
- training the model in fit().
- plotting the losses, visualize the predictions on train and test.
- track hyperparameter changes in the training process.

Focus on init(), forward(), and fit() to understand the pipeline. 
'''
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.decomposition import PCA
from torch import nn
import matplotlib.pyplot as plt
from simulator import LinearDynamics, SpikeSimulator
import torch.nn.functional as F
from load import *
import torch.optim.lr_scheduler as lr_scheduler
import itertools
import os
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import colormaps

class IdentifiedPCA(torch.utils.data.Dataset):
    """Dataset class for PCA latent dynamics training using neurons aligned with connectome"""
    def __init__(self, latent_dim=5, zoomin_factor=5, sequence_length=50, test_seq_length=50, 
                 sheet_name=0, data_dir='data/activity/WT_NoStim', connectome_path='data/connectome/White1986', detrend = True):
        """Initialize the IdentifiedPCA Dataset with both training and test windows"""
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            ('mps' if torch.backends.mps.is_available() else 'cpu'))
        

        self.full_traces, _, self.neuron_ids, self.fps, self.sim_to_dataset_idx = load_identified_data(
            data_dir=data_dir, 
            sheet_name=sheet_name,
            connectome_path=connectome_path,
            detrend=detrend
        )
        
        # Store the number of neurons
        self.n_neurons = self.full_traces.shape[0]
        
        # Store sequence lengths
        self.sequence_length = sequence_length
        self.test_seq_length = test_seq_length
        self.total_seq_length = sequence_length + test_seq_length
        
        # Ensure we have enough data for both train and test
        if self.full_traces.shape[1] < self.total_seq_length:
            print(f"Warning: Available data ({self.full_traces.shape[1]} frames) is less than requested "
                  f"({self.total_seq_length} frames). Using all available data.")
            self.total_seq_length = self.full_traces.shape[1]
            self.test_seq_length = max(0, self.total_seq_length - self.sequence_length)
        
        
        # Apply PCA to the filtered neural activity data
        pca = PCA(n_components=min(latent_dim, self.n_neurons), svd_solver='full')
        latent_traces_np = pca.fit_transform(self.full_traces.cpu().numpy().T)
        self.latent_traces = torch.tensor(latent_traces_np.T, dtype=torch.float32, device=device)
        self.latent_dim = self.latent_traces.shape[0]  # May be less than requested if fewer overlapping neurons
        
        # Store PCA components and the PCA object for later use
        self.pca = pca
        self.components = pca.components_  # Shape (latent_dim, n_neurons)
        
        # Calculate simulation parameters
        self.zoomin_factor = zoomin_factor
        self.points_per_frame = zoomin_factor
        
        # Training data parameters
        self.train_sim_steps = sequence_length * zoomin_factor
        
        # Test data parameters
        self.test_sim_steps = test_seq_length * zoomin_factor

        # Set up initial state x0 from the first timepoint
        self.x0 = torch.zeros(1, self.n_neurons, dtype=torch.float32, device=device)
        for i in range(self.n_neurons):
            self.x0[0, i] = self.full_traces[i, 0]
        
        # Set up initial state for test data (starts at the end of training sequence)
        self.test_x0 = torch.zeros(1, self.n_neurons, dtype=torch.float32, device=device)
        if test_seq_length > 0:
            for i in range(self.n_neurons):
                self.test_x0[0, i] = self.full_traces[i, sequence_length]
        
        # Create expanded latent traces for training
        self.train_latent = self._create_expanded_latent(0, sequence_length, self.train_sim_steps)
        self.train_weights = self._create_weights(sequence_length, self.train_sim_steps)
        
        # Create expanded latent traces for testing
        if test_seq_length > 0:
            self.test_latent = self._create_expanded_latent(sequence_length, test_seq_length, self.test_sim_steps)
            self.test_weights = self._create_weights(test_seq_length, self.test_sim_steps)
        else:
            self.test_latent = None
            self.test_weights = None
        
        # For backward compatibility
        self.latent = self.train_latent
        self.train_traces = self._create_expanded_traces(0, sequence_length, self.train_sim_steps)
        self.weights = self.train_weights
        self.sim_steps = self.train_sim_steps
    
    def _create_expanded_latent(self, start_idx, length, sim_steps):
        """Create expanded latent traces for a given window"""
        end_idx = start_idx + length
        window_traces = self.latent_traces[:, start_idx:end_idx]
        
        expanded_latent = torch.zeros(self.latent_dim, sim_steps, device=window_traces.device)
        actual_length = min(length, window_traces.shape[1])
        
        # Only populate at exact frame intervals
        for i in range(actual_length):
            idx = i * self.points_per_frame
            if idx < sim_steps:
                expanded_latent[:, idx] = window_traces[:, i]
        
        return expanded_latent
    
    def _create_expanded_traces(self, start_idx, length, sim_steps):
        """Create expanded latent traces for a given window"""
        end_idx = start_idx + length
        window_traces = self.full_traces[:, start_idx:end_idx]
        
        expanded_traces = torch.zeros(self.n_neurons, sim_steps, device=window_traces.device)
        actual_length = min(length, window_traces.shape[1])
        
        # Only populate at exact frame intervals
        for i in range(actual_length):
            idx = i * self.points_per_frame
            if idx < sim_steps:
                expanded_traces[:, idx] = window_traces[:, i]
        
        return expanded_traces
    
    def _create_weights(self, length, sim_steps):
        """Create weight matrix for a given window"""
        weights = torch.zeros(self.latent_dim, sim_steps, device=self.latent_traces.device)
        
        # Only set weights at exact frame intervals
        for i in range(length):
            idx = i * self.points_per_frame
            if idx < sim_steps:
                weights[:, idx] = 1.0
        
        return weights
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.train_latent, self.x0, self.train_weights, self.fps, self.train_traces

class TrialPCA(nn.Module):
    """
    Trial-specific PCA model that simulates dynamics only for identified neurons
    and projects them directly to latent space.
    """
    def __init__(self,
                 latent_dim=5,
                 beta=1.0,
                 tau=5.0,
                 zoomin_factor=5,
                 sheet_name=0,
                 data_dir='data/activity/WT_NoStim',
                 connectome_path='data/connectome/White1986',
                 sequence_length=50,
                 test_seq_length= 10,
                 detrend = True,
                 noise_strength = 0,
                 synaptic_gain = 1.0,
                 constrained = True,
                 train_gap_junctions = False):
        """Initialize the TrialPCA model for a specific trial"""
        super().__init__()
        
        # Create device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                ('mps' if torch.backends.mps.is_available() else 'cpu'))
        
        # Create dataset to get PCA components and data
        self.dataset = IdentifiedPCA(
            latent_dim=latent_dim,
            zoomin_factor=zoomin_factor,
            sequence_length=sequence_length,
            test_seq_length=test_seq_length,
            sheet_name=sheet_name,
            data_dir=data_dir,
            connectome_path=connectome_path,
            detrend= detrend
        )
        
        # Store parameters
        self.latent_dim = latent_dim
        self.n_neurons = self.dataset.n_neurons
        self.fps = self.dataset.fps
        self.zoomin_factor = zoomin_factor
        self.sheet_name = sheet_name
        self.sequence_length = sequence_length
        self.constrained = constrained
        self.train_gap_junctions = train_gap_junctions
        # Create a dynamics model for all neurons in the connectome
        self.dynamics_model = LinearDynamics(
            connectome_path=connectome_path,
            beta=beta,
            tau=tau,
            zoomin_factor=zoomin_factor,
            sheet_name=sheet_name,
            noise_strength=noise_strength,
            synaptic_gain_g=synaptic_gain,
            constrained = constrained,
            train_gap_junctions = train_gap_junctions
        )
        
        # Calculate alpha based on fps and zoomin_factor
        self.alpha = 1.0 / (self.zoomin_factor * self.fps)
        
        # The dynamics model uses all neurons from the connectome
        self.n_sim_neurons = self.dynamics_model.n_units
        
        # Create neuron mapping between dataset and simulation
        self.create_neuron_mapping(connectome_path)
        
        # Create projection matrix from simulation size to latent dimensions with masking
        self.projection = nn.Linear(self.n_sim_neurons, self.latent_dim, bias=True).to(self.device)
        self.observed_projection = nn.Linear(self.n_neurons, self.latent_dim, bias=True).to(self.device)

        
        # Initialize projection with masked PCA components
        self.initialize_projection()
    
    def forward(self, x0, t_span=None, alpha=None, time_steps=None):
        """Simulate dynamics and project to latent space using masked projection"""
        # Set default time span and alpha if not provided
        if t_span is None:
            t_span = (0, self.sequence_length / self.fps)
        
        if alpha is None:
            alpha = self.alpha
            
        if time_steps is None:
            time_steps = self.dataset.sim_steps
        
        # Run neural simulation for all neurons in the connectome
        t, neural_states = self.dynamics_model.simulate(
            t_span=t_span, 
            x0=x0, 
            alpha=alpha,
            time_steps=time_steps
        )
        
        # 2. MODIFIED: Extract observed states (M neurons) using pre-ordered indices
        observed_neural_states = neural_states[:, :, self.sim_indices_for_observed]

        # 3. MODIFIED: Project observed states (M -> L) using the new layer
        time_steps_actual = observed_neural_states.shape[0]
        batch_size = observed_neural_states.shape[1]

        latent_states = self.observed_projection(
            observed_neural_states.reshape(-1, self.n_neurons) # Reshape to [T*B, M]
        ).reshape(time_steps_actual, batch_size, self.latent_dim) # Reshape back to [T, B, L]
        
        return t, neural_states ,observed_neural_states, latent_states
    
    def fit(self, n_epochs=100, learning_rate=0.01, weight_decay=1e-5, # Added L2
        lambda_cov=0.1, lambda_corr=0.1, # Added loss weights
        scheduler_patience=10, scheduler_factor=0.1, # Added scheduler params
        checkpoint_path=None, checkpoint_frequency=10, resume_from=None):
        """
        Train the model using MSE reconstruction loss, and evaluate on test set
        """
        torch.autograd.set_detect_anomaly(True)
        
        self.train()
        self.dynamics_model.train()
        
        # Set up data and optimizer
        train_latent = self.dataset.train_latent.to(self.device)
        train_weights = self.dataset.train_weights.to(self.device)
        train_traces = self.dataset.train_traces.to(self.device)
        
        # Create an initial condition with all zeros for the full simulation
        x0 = torch.zeros(1, self.n_sim_neurons, dtype=torch.float32, device=self.device)
        
        # Fill in the values for the overlapping neurons
        for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
            x0[0, sim_idx] = self.dataset.x0[0, dataset_idx]
        
        # Prepare test initial conditions if test data exists
        have_test_data = self.dataset.test_seq_length > 0
        if have_test_data:
            test_x0 = torch.zeros(1, self.n_sim_neurons, dtype=torch.float32, device=self.device)
            for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
                test_x0[0, sim_idx] = self.dataset.test_x0[0, dataset_idx]
            
            test_latent = self.dataset.test_latent.to(self.device)
            test_weights = self.dataset.test_weights.to(self.device)
        
        params = list(self.dynamics_model.parameters()) + list(self.observed_projection.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay) # Added weight_decay
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                factor=scheduler_factor,
                                                patience=scheduler_patience,
                                                verbose=True) # Added scheduler
        
        # --- Calculate Actual and "Before Training" Covariance Matrices ---
        actual_neural_cov = self._calculate_neural_covariance(
            train_traces,
            train_weights[0, :] # Use weights from one latent dim for timepoints
        )

        sim_neural_cov_before_training = None
        if self.n_neurons > 0 : # Only if there are observed neurons
            self.eval()
            self.dynamics_model.eval()
            with torch.no_grad():
                train_t_span_initial = (0, self.sequence_length / self.fps)
                _, _, initial_sim_observed_neural_states_Tsim_B_M, _ = self.forward(
                    x0=x0,
                    t_span=train_t_span_initial,
                    time_steps=self.dataset.train_sim_steps
                )
                # initial_sim_observed_neural_states is [T_sim, B, M], need [M, T_sim]
                initial_sim_neural_M_Tsim = initial_sim_observed_neural_states_Tsim_B_M.squeeze(1).t()
                sim_neural_cov_before_training = self._calculate_neural_covariance(
                    initial_sim_neural_M_Tsim,
                    train_weights[0, :]
                )
            self.train() # Switch back to train mode
            self.dynamics_model.train()
        else:
            sim_neural_cov_before_training = np.zeros((0,0))
        
        # Initialize training history
        start_epoch = 0
        train_losses_mse = []
        train_losses_cov = []   # ADDED
        train_losses_corr = []  # ADDED
        test_losses_mse = []
        test_cov_similarities = []
        train_losses_neural_mse = []
        train_cov_similarities = []
        # --- ADDED: Parameter history dictionary ---
        param_history = {
            'beta': [], 'tau': [], 'd_mean': [], 'alpha_cubic_mean': [],
            'beta_cubic_mean': [], 'gamma_cubic_mean': [], # Assuming gamma_cubic exists in dynamics_model
            'A_weight_norm': [], 'proj_weight_norm': []
        }
        sim_observed_neural_states_final_M_Tsim = None
        # --- End Added ---
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            print(f"Attempting to resume training from checkpoint: {resume_from}")
            try:
                loaded_epoch, loaded_train_mse, loaded_test_mse, loaded_cov_sim = self.load_checkpoint(
                    resume_from, optimizer, scheduler # Pass optimizer and scheduler to load their states
                )

                start_epoch = loaded_epoch
                map_location = self.device
                checkpoint = torch.load(resume_from, map_location=map_location) # Reload temp to get history
                train_losses_mse = checkpoint.get('train_losses_mse', [])
                train_losses_cov = checkpoint.get('train_losses_cov', [])
                train_losses_corr = checkpoint.get('train_losses_corr', [])
                test_losses_mse = checkpoint.get('test_losses_mse', [])
                test_cov_similarities = checkpoint.get('test_cov_similarities', [])
                train_cov_similarities = checkpoint.get('train_cov_similarities', []) # <<< ADDED Load
                param_history = checkpoint.get('param_history', param_history) # Load param history

                print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Warning: Could not load checkpoint '{resume_from}': {e}. Starting fresh training.")
                start_epoch = 0
                train_losses_mse = []
                test_losses_mse = []
                test_cov_similarities = []
                train_cov_similarities = []
        
        print(f"Starting training for {n_epochs} epochs...")
        train_t_span = (0, self.sequence_length / self.fps)
        
        if have_test_data:
            test_t_span = (0, self.dataset.test_seq_length / self.fps)
        
        for epoch in range(start_epoch, n_epochs):
            optimizer.zero_grad()
            
            try:
                # Training phase
                self.train()
                self.dynamics_model.train()
                
                # Forward pass with exact simulation steps
                _, _, simulated_traces, latent_states = self.forward(
                    x0=x0,
                    t_span=train_t_span,
                    time_steps=self.dataset.train_sim_steps
                )
                
                # Check for NaN values
                if torch.isnan(latent_states).any():
                    print(f"\nNaN detected in training sim at epoch {epoch}")
                    continue
                
                # Calculate MSE loss - comparing latent_states to target_latent
                # Reshape latent_states from [T, 1, D] to [D, T]
                pred_latent = latent_states.squeeze(1).t()
                sim_observed_neural_M_Tsim = simulated_traces.squeeze(1).t() # [M, T_sim]
                # Ensure weights have compatible shape [L, T_sim]
                if train_weights.dim() == 1:
                    w = train_weights.unsqueeze(0).expand_as(train_latent)
                else:
                    w = train_weights
                neural_weights = w[0, :].unsqueeze(0).expand_as(train_traces)
                # 1. Weighted MSE Loss
                #mse_loss = torch.sum(w * (pred_latent - train_latent) ** 2) / torch.sum(w).clamp(min=1.0)
                mse_loss = torch.sum(w * (pred_latent - train_latent) ** 2) / torch.sum(w).clamp(min=1.0)

                # 2. Covariance Loss (Frobenius norm of difference) # ADDED
                cov_loss = torch.tensor(0.0, device=self.device)
                if lambda_cov > 0 and self.latent_dim > 1: # Need >1 dim for covariance
                    try:
                        # Ensure inputs to weighted_covariance are [L, T]
                        cov_loss = weighted_covariance(sim_observed_neural_M_Tsim, train_traces, neural_weights)
                    except Exception as e:
                        print(f"Warning: Could not calculate covariance loss at epoch {epoch}: {e}")
                        cov_loss = torch.tensor(0.0, device=self.device) # Assign zero loss if error


                # 3. Correlation Loss (Average 1 - Pearson Corr) # ADDED
                corr_loss = torch.tensor(0.0, device=self.device)
                if lambda_corr > 0:
                    try:
                        # Ensure inputs to weighted_pearson_correlation are [L, T]
                        corr_loss = weighted_pearson_correlation(pred_latent, train_latent, w)
                    except Exception as e:
                        print(f"Warning: Could not calculate correlation loss at epoch {epoch}: {e}")
                        corr_loss = torch.tensor(1.0, device=self.device) # Assign max loss (1.0) if error


                # 4. Total Loss # MODIFIED
                total_loss = mse_loss + lambda_cov * cov_loss + lambda_corr * corr_loss


                self.eval()
                self.dynamics_model.eval()
                with torch.no_grad(): 
                    _, _, simulated_traces, latent_states = self.forward(
                        x0=x0,
                        t_span=train_t_span,
                        time_steps=self.dataset.train_sim_steps
                    )
                    neural_weights = w[0, :].unsqueeze(0).expand_as(train_traces) #[M, T_sim]
                    simulated_traces = simulated_traces.squeeze(1).t()
                    # Calculate weighted neural MSE
                    current_train_neural_mse = (torch.sum(neural_weights * (simulated_traces - train_traces)**2) /
                                                torch.sum(neural_weights).clamp(min=1.0)).item()

                    weight_indices = torch.where(neural_weights[0] > 0)[0] # Get indices for actual frames
                    if len(weight_indices) > 1 and self.n_neurons > 0: # Need >1 point and >0 neurons
                        if simulated_traces is not None:
                            sim_cov_train = self._calculate_neural_covariance(simulated_traces, neural_weights[0, :])
                        else:
                            print("simulated traces is None!")
                        actual_cov_train = self._calculate_neural_covariance(train_traces, neural_weights[0, :])

                        data_tensor = torch.from_numpy(actual_cov_train).float().to(self.device)
                        sim_tensor = torch.from_numpy(sim_cov_train).float().to(self.device)

                        # Calculate cosine similarity between flattened covariance matrices
                        sim_cov_flat_train = sim_tensor.view(-1)
                        actual_cov_flat_train = data_tensor.view(-1)
                        current_train_cov_sim = F.cosine_similarity(sim_cov_flat_train.unsqueeze(0), actual_cov_flat_train.unsqueeze(0), eps=1e-6).item()


                # Check for NaN/Inf in loss components
                if torch.isnan(mse_loss) or torch.isinf(mse_loss): print(f"NaN/Inf in MSE loss epoch {epoch}")
                if torch.isnan(cov_loss) or torch.isinf(cov_loss): print(f"NaN/Inf in Cov loss epoch {epoch}")
                if torch.isnan(corr_loss) or torch.isinf(corr_loss): print(f"NaN/Inf in Corr loss epoch {epoch}")
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN or Inf detected in total loss at epoch {epoch}. Skipping backward pass.")
                    continue # Skip backprop and update
                

                # --- Backward Pass and Optimization --- # (Error handling added)
                try:
                    total_loss.backward()
                    # Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()


                except Exception as e:
                    print(f"Error during backward pass or optimizer step in epoch {epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue # Skip epoch if backward/step fails


                # --- Log Training Losses --- # (Modified to log new losses)
                train_losses_mse.append(mse_loss.item())
                train_losses_cov.append(lambda_cov * cov_loss.item() if isinstance(cov_loss, torch.Tensor) else lambda_cov * cov_loss)
                train_losses_corr.append(corr_loss.item() if isinstance(corr_loss, torch.Tensor) else corr_loss)
                train_losses_neural_mse.append(current_train_neural_mse) # Append the calculated metric
                train_cov_similarities.append(current_train_cov_sim)
                with torch.no_grad():
                    param_history['beta'].append(self.dynamics_model.beta.item())
                    param_history['tau'].append(self.dynamics_model.tau.item())
                    param_history['d_mean'].append(self.dynamics_model.d.mean().item())
                    param_history['alpha_cubic_mean'].append(self.dynamics_model.alpha_cubic.mean().item())
                    param_history['beta_cubic_mean'].append(self.dynamics_model.beta_cubic.mean().item())
                    # Add gamma_cubic if it exists in your LinearDynamics model
                    if hasattr(self.dynamics_model, 'gamma_cubic'):
                        param_history['gamma_cubic_mean'].append(self.dynamics_model.gamma_cubic.mean().item())
                    else: # Handle case where gamma_cubic might not exist (append NaN or default)
                        # Ensure key exists if we loaded history that had it
                        if 'gamma_cubic_mean' not in param_history: param_history['gamma_cubic_mean'] = []
                        param_history['gamma_cubic_mean'].append(np.nan) # Or some placeholder

                    param_history['A_weight_norm'].append(torch.norm(self.dynamics_model.A.weight).item())
                    param_history['proj_weight_norm'].append(torch.norm(self.observed_projection.weight).item())

                
                # Apply connectivity constraints
                with torch.no_grad():
                    if self.constrained:
                        self.dynamics_model.A.weight *= self.dynamics_model.A_mask
                if epoch == n_epochs - 1:
                     sim_observed_neural_states_final_M_Tsim = sim_observed_neural_M_Tsim.detach().clone()
                
                
                # Test phase
                if have_test_data:
                    self.eval()
                    self.dynamics_model.eval()
                    
                    with torch.no_grad():
                        # Forward pass on test data
                        _, neural_states, _, latent_states = self.forward(
                            x0=test_x0,
                            t_span=test_t_span,
                            time_steps=self.dataset.test_sim_steps
                        )
                        
                        # Calculate MSE loss on test data
                        pred_latent = latent_states.squeeze(1).t()
                        test_mse_loss = torch.sum(test_weights * (pred_latent - test_latent) ** 2) / torch.sum(test_weights)
                        test_losses_mse.append(test_mse_loss.item())
                        
                        # Calculate covariance similarity
                        # Get actual neural activity from dataset
                        actual_neural = self.dataset.full_traces[:, self.dataset.sequence_length:
                                                               self.dataset.sequence_length + self.dataset.test_seq_length]
                        
                        # Extract simulated activity for overlapping neurons
                        sim_neural = torch.zeros_like(actual_neural, device=self.device)
                        
                        # Sample the neural states at the original time points (not expanded ones)
                        sample_indices = torch.linspace(0, neural_states.shape[0]-1, actual_neural.shape[1]).long()
                        
                        # Extract simulated activity for each neuron - now directly aligned with dataset
                        for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
                            sim_neural[dataset_idx] = neural_states[sample_indices, 0, sim_idx]
                        
                        # No need to permute rows for covariance calculation since they're already aligned
                        
                        # Calculate covariance matrices (neurons x neurons)
                        sim_cov = self._calculate_neural_covariance(sim_neural, test_weights[0, :])
                        actual_cov = self._calculate_neural_covariance(actual_neural, test_weights[0, :])

                        data_tensor = torch.from_numpy(actual_cov).float().to(self.device)
                        sim_tensor = torch.from_numpy(sim_cov).float().to(self.device)

                        # Calculate cosine similarity between flattened covariance matrices
                        sim_cov_flat = sim_tensor.view(-1)
                        actual_cov_flat = data_tensor.view(-1)
                        
                        cos_sim = F.cosine_similarity(sim_cov_flat.unsqueeze(0), actual_cov_flat.unsqueeze(0)).item()
                        test_cov_similarities.append(cos_sim)
                # --- Scheduler Step --- # ADDED
                # Step the scheduler based on test MSE loss if available, otherwise train MSE loss
                metric_for_scheduler = test_losses_mse[-1] if have_test_data and not np.isnan(test_losses_mse[-1]) else mse_loss.item()
                if not np.isnan(metric_for_scheduler):
                    scheduler.step(metric_for_scheduler)
                else:
                    print(f"Warning: Metric for LR scheduler is NaN at epoch {epoch}. Skipping scheduler step.")

                # Print progress
                if epoch % 1 == 0 or epoch == n_epochs - 1:
                    if have_test_data:
                        print(f"Epoch {epoch+1}/{n_epochs}, Train MSE: {mse_loss.item():.3f}, Train Cov Sim: {current_train_cov_sim:.3f}, "
                              f"Test MSE: {test_losses_mse[-1]:.3f}, Tes Cov Sim: {test_cov_similarities[-1]:.3f}")
                    else:
                        print(f"Epoch {epoch+1}/{n_epochs}, Train MSE: {mse_loss.item():.6f}")
            
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
                # --- Calculate "After Training" Neural Covariance Matrix ---
        
        sim_neural_cov_after_training = None
        if self.n_neurons > 0:
            if sim_observed_neural_states_final_M_Tsim is not None: # Use from last epoch if available
                 sim_neural_cov_after_training = self._calculate_neural_covariance(
                    sim_observed_neural_states_final_M_Tsim,
                    train_weights[0, :]
                )
            else: # Fallback: run a fresh forward pass if last epoch's states weren't stored (e.g., due to error or n_epochs=0)
                self.eval()
                self.dynamics_model.eval()
                with torch.no_grad():
                    train_t_span_final = (0, self.sequence_length / self.fps)
                    _, _, final_sim_obs_neural_Tsim_B_M, _ = self.forward(
                        x0=x0,
                        t_span=train_t_span_final,
                        time_steps=self.dataset.train_sim_steps
                    )
                    final_sim_neural_M_Tsim = final_sim_obs_neural_Tsim_B_M.squeeze(1).t()
                    sim_neural_cov_after_training = self._calculate_neural_covariance(
                        final_sim_neural_M_Tsim,
                        train_weights[0, :]
                    )
        else:
            sim_neural_cov_after_training = np.zeros((0,0))

        # Save final checkpoint
        if checkpoint_path:
            self.save_checkpoint(
                f"{checkpoint_path}_final.pt",
                n_epochs,
                train_losses_mse, train_losses_cov, train_losses_corr, train_losses_neural_mse, train_cov_similarities,
                test_losses_mse, test_cov_similarities,
                optimizer, scheduler,
                param_history
            )
        
        heatmap_save_path = None
        if checkpoint_path:
            heatmap_save_path = f"{checkpoint_path}_constrained:{self.constrained}_neural_cov_heatmaps.png"
        elif self.sheet_name is not None : # Fallback naming if no checkpoint_path
             heatmap_save_path = f"figs/trial_{self.sheet_name}_constrained:{self.constrained}_neural_cov_heatmaps.png"
             Path(heatmap_save_path).parent.mkdir(parents=True, exist_ok=True)


        self.plot_covariance_heatmaps(
            actual_neural_cov,
            sim_neural_cov_before_training,
            sim_neural_cov_after_training,
            save_path=heatmap_save_path
        )
        
        # --- Plot Final Losses --- # Modified to pass new losses
        self.plot_loss(train_losses_mse, train_losses_cov, train_losses_corr, # Pass new losses
                    test_losses_mse, train_losses_neural_mse, train_cov_similarities, test_cov_similarities,
                    save_path=f"{checkpoint_path}_constrained:{self.constrained}_loss.png" if checkpoint_path else None)
        
        self.plot_parameter_evolution(
            param_history,
            save_path=f"{checkpoint_path}_constrained:{self.constrained}_params.png" if checkpoint_path else None
        )

        return train_losses_mse, test_losses_mse, test_cov_similarities # Return original metrics for consistency

# -------------------VISUALIZATION & CHECKPOINT METHODS-------------------
    def save_checkpoint(self, path, epoch, train_losses_mse, train_losses_cov, train_losses_corr, train_losses_neural_mse, train_cov_similarities, # Added new losses
                        test_losses_mse, test_cov_similarities, optimizer, scheduler, param_history): # Added scheduler
        """Save training checkpoint with new loss terms and scheduler state"""
        checkpoint_data = {
            'epoch': epoch,
            'dynamics_model_state': self.dynamics_model.state_dict(),
            'observed_projection_state': self.observed_projection.state_dict(), # Use correct name
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
            # Save all loss histories
            'train_losses_mse': train_losses_mse,
            'train_losses_cov': train_losses_cov,     # ADDED
            'train_losses_corr': train_losses_corr,   # ADDED
            'train_losses_neural_mse': train_losses_neural_mse,
            'train_cov_similarities': train_cov_similarities,
            'test_losses_mse': test_losses_mse,
            'test_cov_similarities': test_cov_similarities,
            # Model architecture info
            'latent_dim': self.latent_dim,
            'n_neurons': self.n_neurons, # Identified neurons
            'n_sim_neurons': self.n_sim_neurons, # Total sim neurons
            'dataset_to_sim_idx': self.dataset_to_sim_idx, # Mapping used
            'param_history': param_history
        }
        # Handle cases where projection might have been loaded with old key 'projection_state'
        # Ensure we save with the correct current key 'observed_projection_state'
        if 'projection_state' in checkpoint_data:
            del checkpoint_data['projection_state']


        torch.save(checkpoint_data, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def load_checkpoint(self, path, optimizer=None, scheduler=None): # Added scheduler
        """Load training checkpoint including scheduler state"""
        # Ensure map_location is set correctly
        map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)

        self.dynamics_model.load_state_dict(checkpoint['dynamics_model_state'])

        # Handle loading observed_projection state (with fallback)
        if 'observed_projection_state' in checkpoint:
            self.observed_projection.load_state_dict(checkpoint['observed_projection_state'])
        elif 'projection_state' in checkpoint:
            print("Warning: Loading 'projection_state' into 'observed_projection_state'.")
            self.observed_projection.load_state_dict(checkpoint['projection_state'])
        else:
            print("Warning: No projection state found in checkpoint.")


        # Load optimizer and scheduler states if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded.")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint: # ADDED check
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # ADDED load
                print("Scheduler state loaded.")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")

        # Load parameters and mappings
        if 'dataset_to_sim_idx' in checkpoint: self.dataset_to_sim_idx = checkpoint['dataset_to_sim_idx']
        if 'n_sim_neurons' in checkpoint: self.n_sim_neurons = checkpoint['n_sim_neurons']
        if 'n_neurons' in checkpoint: self.n_neurons = checkpoint['n_neurons']
        if 'latent_dim' in checkpoint: self.latent_dim = checkpoint['latent_dim']

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")

        # Load loss history
        train_losses_mse = checkpoint.get('train_losses_mse', [])
        train_losses_cov = checkpoint.get('train_losses_cov', [])   # ADDED load
        train_losses_corr = checkpoint.get('train_losses_corr', []) # ADDED load
        train_losses_neural_mse = checkpoint.get('train_losses_neural_mse', [])
        train_cov_similarities = checkpoint.get('train_cov_similarities', [])
        test_losses_mse = checkpoint.get('test_losses_mse', [])
        test_cov_similarities = checkpoint.get('test_cov_similarities', [])
        # --- Load Parameter History ---
        # Define the expected keys for the current default history structure
        default_param_history_keys = [
             'beta', 'tau', 'd_mean', 'alpha_cubic_mean',
             'beta_cubic_mean', 'gamma_cubic_mean',
             'A_weight_norm', 'proj_weight_norm'
        ]
        default_param_history = {key: [] for key in default_param_history_keys}
        # Load from checkpoint, using the default structure if key is missing
        param_history = checkpoint.get('param_history', default_param_history)
        # Ensure all expected keys are present in the loaded history, adding empty lists if not
        for key in default_param_history_keys:
             if key not in param_history:
                 print(f"Warning: Parameter history key '{key}' not found in checkpoint. Initializing empty.")
                 param_history[key] = []

        # Return epoch and primary loss histories
        return checkpoint['epoch'], train_losses_mse, test_losses_mse, test_cov_similarities

    def plot_loss(self, train_losses_mse, train_losses_cov, train_losses_corr, # Added new losses
                test_losses=None, train_losses_neural_mse=None, train_cov_similarities=None, cov_similarities=None, save_path=None):
        """Plot training and test loss history, including breakdown of train loss"""
        num_plots = 1
        if test_losses and not all(np.isnan(l) for l in test_losses): num_plots += 1
        if cov_similarities and not all(np.isnan(s) for s in cov_similarities): num_plots +=1
        if train_losses_neural_mse and not all(np.isnan(t) for t in train_losses_neural_mse): num_plots +=1 
        plot_train_neural_mse = train_losses_neural_mse and not all(np.isnan(l) for l in train_losses_neural_mse)
        plot_train_cov_sim = train_cov_similarities and not all(np.isnan(s) for s in train_cov_similarities) # <<< Check train cov sim
        plot_train_neural_metrics = plot_train_neural_mse or plot_train_cov_sim # <<< Combine check

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes] # Make it iterable if only one plot

        plot_idx = 0

        # --- Plot Training Loss Components ---
        ax_train = axes[plot_idx]
        epochs = range(1, len(train_losses_mse) + 1)
        ax_train.plot(epochs, train_losses_mse, 'b-', alpha=0.8, label='Train MSE')
        # Only plot cov/corr if they were used (check length > 0) # ADDED plot logic
        if train_losses_cov:
            ax_train.plot(epochs, train_losses_cov, 'g-', alpha=0.6, label='Train Cov Loss (scaled)')

        # Use log scale for potentially large differences, but handle zeros/negatives # ADDED robustness
        try:
            ax_train.set_yscale('log')
            # Set a minimum value for the y-axis if using log scale
            all_train_losses = [l for l_list in [train_losses_mse, train_losses_cov, train_losses_corr] for l in l_list if l > 0]
            if all_train_losses:
                min_y = min(all_train_losses) / 10
                ax_train.set_ylim(bottom=max(min_y, 1e-7)) # Avoid zero or negative limits
            else:
                ax_train.set_ylim(bottom=1e-7) # Default bottom if no positive losses
        except ValueError: # Handle cases where all values might be zero or negative
            pass # Keep linear scale
        ax_train.set_ylabel('Training Loss Components')
        ax_train.set_title('Training Loss History')
        ax_train.grid(True, alpha=0.3)
        ax_train.legend()
        plot_idx += 1

        # --- Plot Test MSE Loss --- (Modified to handle NaNs better)
        if test_losses and not all(np.isnan(l) for l in test_losses):
            ax_test_mse = axes[plot_idx]
            # Filter out NaNs for plotting
            valid_indices_mse = [i for i, x in enumerate(test_losses) if not np.isnan(x)]
            if valid_indices_mse: # Check if there are any valid points
                valid_epochs_mse = [epochs[i] for i in valid_indices_mse]
                valid_losses_mse = [test_losses[i] for i in valid_indices_mse]
                ax_test_mse.plot(valid_epochs_mse, valid_losses_mse, 'r-', alpha=0.8, label='Test MSE (temporal)')
                ax_test_mse.set_ylabel('Test MSE Loss')
                ax_test_mse.set_title('Test MSE History')
                ax_test_mse.grid(True, alpha=0.3)
                ax_test_mse.legend()
                # Consider log scale for test MSE as well
                try:
                    ax_test_mse.set_yscale('log')
                    positive_test_losses = [l for l in valid_losses_mse if l > 0]
                    if positive_test_losses:
                        min_y_test = min(positive_test_losses) / 10
                        ax_test_mse.set_ylim(bottom=max(min_y_test, 1e-7))
                    else:
                        ax_test_mse.set_ylim(bottom=1e-7) # Default if no positive losses
                except ValueError:
                    pass
            plot_idx += 1

            # --- Plot Train Neural Metrics --- <<< MODIFIED PLOT BLOCK
            if plot_train_neural_metrics:
                ax_train_neu = axes[plot_idx]
                ax_train_neu.set_ylabel('Train Neural MSE', color='c')
                ax_train_neu.set_title('Train Neural Metrics (MSE & Cov Sim)')
                ax_train_neu.grid(True, alpha=0.3, axis='y', linestyle=':') # Grid for primary y-axis

                # Plot Train Neural MSE if available
                if plot_train_neural_mse:
                    valid_indices_neu = [i for i, x in enumerate(train_losses_neural_mse) if not np.isnan(x)]
                    if valid_indices_neu:
                        valid_epochs_neu = [epochs[i] for i in valid_indices_neu]
                        valid_losses_neu = [train_losses_neural_mse[i] for i in valid_indices_neu]
                        p1, = ax_train_neu.plot(valid_epochs_neu, valid_losses_neu, 'c-', alpha=0.8, label='Train MSE (Neural)')
                        try: ax_train_neu.set_yscale('log') # Try log scale for MSE
                        except ValueError: pass
                        ax_train_neu.tick_params(axis='y', labelcolor='c')

                # Plot Train Cov Sim on secondary axis if available
                if plot_train_cov_sim:
                    ax_train_cov = ax_train_neu.twinx() # Create secondary y-axis
                    ax_train_cov.set_ylabel('Train Cov Sim', color='darkorange')
                    valid_indices_train_cov = [i for i, x in enumerate(train_cov_similarities) if not np.isnan(x)]
                    if valid_indices_train_cov:
                        valid_epochs_train_cov = [epochs[i] for i in valid_indices_train_cov]
                        valid_sims_train_cov = [train_cov_similarities[i] for i in valid_indices_train_cov]
                        p2, = ax_train_cov.plot(valid_epochs_train_cov, valid_sims_train_cov, '-', color='darkorange', alpha=0.8, label='Train Cov Sim (Neural)')
                        ax_train_cov.set_ylim(-1.05, 1.05) # Cov sim range
                        ax_train_cov.tick_params(axis='y', labelcolor='darkorange')

                # Add combined legend
                lines = []
                if plot_train_neural_mse and 'p1' in locals(): lines.append(p1)
                if plot_train_cov_sim and 'p2' in locals(): lines.append(p2)
                if lines: ax_train_neu.legend(lines, [l.get_label() for l in lines])

                plot_idx += 1
            # --- End Modified Plot Block ---


        # --- Plot Test Covariance Similarity --- (Modified to handle NaNs better)
        if cov_similarities and not all(np.isnan(s) for s in cov_similarities):
            ax_cov_sim = axes[plot_idx]
            # Filter out NaNs
            valid_indices_cov = [i for i, x in enumerate(cov_similarities) if not np.isnan(x)]
            if valid_indices_cov: # Check if there are any valid points
                valid_epochs_cov = [epochs[i] for i in valid_indices_cov]
                valid_sims_cov = [cov_similarities[i] for i in valid_indices_cov]
                ax_cov_sim.plot(valid_epochs_cov, valid_sims_cov, 'g-', alpha=0.8, label='Test Neural Cov Sim')
                ax_cov_sim.set_ylabel('Cosine Similarity')
                ax_cov_sim.set_title('Test Neural Covariance Similarity')
                ax_cov_sim.set_ylim(-1.05, 1.05) # Cosine similarity range
                ax_cov_sim.grid(True, alpha=0.3)
                ax_cov_sim.legend()
            plot_idx += 1


        # Common x-axis label
        axes[-1].set_xlabel('Epoch')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def visualize_prediction_test(self, test_sequence_length=None, save_path=None):
        """Generate and visualize a prediction on the test window"""
        if test_sequence_length is None:
            test_sequence_length = self.dataset.test_seq_length
        
        # Ensure we have test data
        if test_sequence_length <= 0:
            print("No test data available for visualization")
            return None
        
        # Set to evaluation mode
        self.eval()
        self.dynamics_model.eval()
        
        with torch.no_grad():
            # Create initial condition for the full simulation size from test initial point
            test_x0 = torch.zeros(1, self.n_sim_neurons, dtype=torch.float32, device=self.device)
            
            # Fill in values for overlapping neurons using test initial conditions
            for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
                test_x0[0, sim_idx] = self.dataset.test_x0[0, dataset_idx]
            
            # Define time span for test window
            test_t_span = (0, test_sequence_length / self.fps)
            
            # Get real latent traces for test window
            test_start = self.dataset.sequence_length
            test_end = test_start + test_sequence_length
            actual_latent = self.dataset.latent_traces[:, test_start:test_end].cpu().numpy()
            
            # Simulate dynamics on test window
            _, neural_states, _, latent_states = self.forward(
                x0=test_x0, 
                t_span=test_t_span,
                time_steps=self.dataset.test_sim_steps
            )
            
            # Convert results to numpy for plotting
            neural_states_np = neural_states[:, 0].cpu().numpy()
            latent_states_np = latent_states[:, 0].cpu().numpy()
            
            # Create time array for test window
            time_steps = neural_states.shape[0]
            t_np = np.linspace(0, test_t_span[1], time_steps)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        
        # Plot neural trajectories for test window
        ax1 = fig.add_subplot(221)
        
        # Get overlapping neurons for visualization
        overlapping_neurons = []
        for sim_idx, dataset_idx in self.sim_to_dataset_idx.items():
            # Find the neuron name
            for name, idx in self.dataset.neuron_ids.items():
                if idx == dataset_idx:
                    overlapping_neurons.append((name, sim_idx, dataset_idx))
                    if len(overlapping_neurons) >= 3:  # Limit to 3 neurons for clarity
                        break
            if len(overlapping_neurons) >= 3:
                break
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(overlapping_neurons)))
        
        for i, (name, sim_idx, dataset_idx) in enumerate(overlapping_neurons):
            # Plot simulated trajectory
            ax1.plot(t_np, neural_states_np[:, sim_idx], color=colors[i], linestyle='-', 
                    label=f'{name} (sim)')
            
            # Plot actual trajectory from dataset for test window
            actual_neural = self.dataset.full_traces[dataset_idx, test_start:test_end].cpu().numpy()
            actual_times = np.linspace(0, test_t_span[1], len(actual_neural))
            ax1.plot(actual_times, actual_neural, color=colors[i], linestyle='--', 
                    label=f'{name} (data)')
        
        ax1.set_title('Test Window: Neural Trajectories (Simulation vs Data)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Activity')
        ax1.legend()
        
        # Plot latent trajectories for test window
        ax2 = fig.add_subplot(222)
        pc_colors = plt.cm.tab10(np.linspace(0, 1, self.latent_dim))
        
        for i in range(min(3, self.latent_dim)):
            ax2.plot(t_np, latent_states_np[:, i], color=pc_colors[i], linestyle='-', 
                    label=f'PC{i+1} (sim)')
        
        ax2.set_title('Test Window: Latent Trajectories')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Activity')
        ax2.legend()
        
        # Plot 3D latent trajectory for test window (both simulated and actual)
        if self.latent_dim >= 3:
            ax3 = fig.add_subplot(223, projection='3d')
            
            # Simulated trajectory
            ax3.plot(latent_states_np[:, 0], latent_states_np[:, 1], latent_states_np[:, 2], 
                    color='blue', linestyle='-', label='Simulated')
            
            # Actual trajectory - need to match timepoints
            time_indices = np.linspace(0, len(t_np)-1, actual_latent.shape[1]).astype(int)
            if len(time_indices) >= 3:
                # Plot actual trajectory
                ax3.plot(actual_latent[0], actual_latent[1], actual_latent[2], 
                        color='red', linestyle='--', label='Actual')
                
                # Add markers at timepoints for better comparison
                for t_idx in range(0, len(time_indices), max(1, len(time_indices)//5)):
                    # Mark simulated point
                    sim_point = latent_states_np[time_indices[t_idx], :3]
                    ax3.scatter(sim_point[0], sim_point[1], sim_point[2], 
                            color='blue', marker='o', s=30)
                    
                    # Mark actual point
                    act_point = actual_latent[:3, t_idx]
                    ax3.scatter(act_point[0], act_point[1], act_point[2], 
                            color='red', marker='x', s=30)
            
            ax3.set_title('Test Window: 3D Latent Trajectory (Simulated vs Actual)')
            ax3.set_xlabel('PC 1')
            ax3.set_ylabel('PC 2')
            ax3.set_zlabel('PC 3')
            ax3.legend()
        
        # Compare simulated latent space with PCA latent space for test window
        ax4 = fig.add_subplot(224)
        time_indices = np.linspace(0, len(t_np)-1, actual_latent.shape[1]).astype(int)
        
        # Plot latent space comparison
        for i in range(min(3, self.latent_dim)):
            color = pc_colors[i]
            ax4.plot(t_np[time_indices], actual_latent[i], color=color, linestyle='--', 
                    label=f'PC{i+1} (data)')
            ax4.plot(t_np[time_indices], latent_states_np[time_indices, i], color=color, linestyle='-', 
                    label=f'PC{i+1} (sim)')
        
        ax4.set_title('Test Window: True vs. Predicted Latent Trajectories')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Activity')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        # Calculate and print test metrics for evaluation
        if actual_latent.shape[1] > 0:
            # Sample simulated values at actual timepoints
            sampled_latent = latent_states_np[time_indices]
            
            # Calculate MSE
            mse = np.mean((sampled_latent - actual_latent.T)**2)
            
            # Calculate correlation between simulated and actual for each dimension
            corrs = []
            for dim in range(min(3, self.latent_dim)):
                corr = np.corrcoef(sampled_latent[:, dim], actual_latent[dim])[0, 1]
                corrs.append(corr)
            
            print(f"Test window evaluation:")
            print(f"  MSE: {mse:.6f}")
            print(f"  Correlations: {[f'{c:.3f}' for c in corrs]}")
        
        return latent_states

    def visualize_prediction(self, sequence_length=None, save_path=None):
        """Generate and visualize a prediction"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        # Set to evaluation mode
        self.eval()
        self.dynamics_model.eval()
        
        with torch.no_grad():
            # Create initial condition for the full simulation size
            x0 = torch.zeros(1, self.n_sim_neurons, dtype=torch.float32, device=self.device)
            
            # Fill in values for overlapping neurons
            for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
                x0[0, sim_idx] = self.dataset.x0[0, dataset_idx]
            
            # Define time span
            t_span = (0, sequence_length / self.fps)
            
            # Get real latent traces for comparison
            actual_latent = self.dataset.latent_traces[:, :sequence_length].cpu().numpy()
            
            # Simulate dynamics
            _, neural_states, _, latent_states = self.forward(
                x0=x0, 
                t_span=t_span,
                time_steps=self.dataset.sim_steps
            )
            
            # Convert results to numpy for plotting
            neural_states_np = neural_states[:, 0].cpu().numpy()
            latent_states_np = latent_states[:, 0].cpu().numpy()
            
            # Create time array
            time_steps = neural_states.shape[0]
            t_np = np.linspace(0, t_span[1], time_steps)
        
        # Create visualization
        fig = plt.figure(figsize=(18, 12))
        
        # Plot neural trajectories for the top overlapping neurons
        ax1 = fig.add_subplot(221)
        
        # Get overlapping neurons for visualization
        overlapping_neurons = []
        for sim_idx, dataset_idx in self.sim_to_dataset_idx.items():
            # Find the neuron name
            for name, idx in self.dataset.neuron_ids.items():
                if idx == dataset_idx:
                    overlapping_neurons.append((name, sim_idx, dataset_idx))
                    if len(overlapping_neurons) >= 3:  # Limit to 3 neurons for clarity
                        break
            if len(overlapping_neurons) >= 3:
                break
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(overlapping_neurons)))
        
        for i, (name, sim_idx, dataset_idx) in enumerate(overlapping_neurons):
            # Plot simulated trajectory
            ax1.plot(t_np, neural_states_np[:, sim_idx], color=colors[i], linestyle='-', 
                     label=f'{name} (sim)')
            
            # Plot actual trajectory from dataset
            actual_timepoints = min(sequence_length, self.dataset.full_traces.shape[1])
            actual_times = np.linspace(0, t_span[1], actual_timepoints)
            actual_data = self.dataset.full_traces[dataset_idx, :actual_timepoints].cpu().numpy()
            ax1.plot(actual_times, actual_data, color=colors[i], linestyle='--', 
                     label=f'{name} (data)')
        
        ax1.set_title('Neural Trajectories (Simulation vs Data)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Activity')
        ax1.legend()
        
        # Plot latent trajectories
        ax2 = fig.add_subplot(222)
        pc_colors = plt.cm.tab10(np.linspace(0, 1, self.latent_dim))
        
        for i in range(min(3, self.latent_dim)):
            ax2.plot(t_np, latent_states_np[:, i], color=pc_colors[i], linestyle='-', 
                     label=f'PC{i+1} (sim)')
        
        ax2.set_title('Latent Trajectories')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Activity')
        ax2.legend()
        
        # Plot 3D latent trajectory (both simulated and actual)
        if self.latent_dim >= 3:
            ax3 = fig.add_subplot(223, projection='3d')
            
            # Simulated trajectory
            ax3.plot(latent_states_np[:, 0], latent_states_np[:, 1], latent_states_np[:, 2], 
                     color='blue', linestyle='-', label='Simulated')
            
            # Actual trajectory - need to match timepoints
            time_indices = np.linspace(0, len(t_np)-1, actual_latent.shape[1]).astype(int)
            if len(time_indices) >= 3:
                # Get the simulated times that match actual data points
                matched_times = t_np[time_indices]
                
                # Plot actual trajectory
                ax3.plot(actual_latent[0], actual_latent[1], actual_latent[2], 
                         color='red', linestyle='--', label='Actual')
                
                # Add markers at matched timepoints for better comparison
                for t_idx in range(0, len(time_indices), max(1, len(time_indices)//10)):
                    # Mark simulated point
                    sim_point = latent_states_np[time_indices[t_idx], :3]
                    ax3.scatter(sim_point[0], sim_point[1], sim_point[2], 
                               color='blue', marker='o', s=30)
                    
                    # Mark actual point
                    act_point = actual_latent[:3, t_idx]
                    ax3.scatter(act_point[0], act_point[1], act_point[2], 
                               color='red', marker='x', s=30)
            
            ax3.set_title('3D Latent Trajectory (Simulated vs Actual)')
            ax3.set_xlabel('PC 1')
            ax3.set_ylabel('PC 2')
            ax3.set_zlabel('PC 3')
            ax3.legend()
        
        # Compare simulated latent space with PCA latent space
        ax4 = fig.add_subplot(224)
        time_indices = np.linspace(0, len(t_np)-1, actual_latent.shape[1]).astype(int)
        
        # Plot actual vs. predicted latent space (first 3 dimensions with same colors)
        for i in range(min(3, self.latent_dim)):
            color = pc_colors[i]
            ax4.plot(t_np[time_indices], actual_latent[i], color=color, linestyle='--', 
                     label=f'PC{i+1} (data)')
            ax4.plot(t_np[time_indices], latent_states_np[time_indices, i], color=color, linestyle='-', 
                     label=f'PC{i+1} (sim)')
        
        ax4.set_title('True vs. Predicted Latent Trajectories')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Activity')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return latent_states

    def plot_parameter_evolution(self, param_history, save_path=None):
        """
        Plots the evolution of key trainable parameters over epochs.
        """
        epochs = range(1, len(param_history['tau']) + 1) # Get number of epochs from history
        if not epochs:
            print("No parameter history to plot.")
            return

        num_params_to_plot = len(param_history)
        # Adjust layout dynamically (e.g., 2 columns)
        n_cols = 2
        n_rows = (num_params_to_plot + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 3), sharex=True)
        axes_flat = axes.flatten()

        plot_idx = 0
        colors = plt.cm.viridis(np.linspace(0, 1, num_params_to_plot))

        # Plot scalar parameters
        if 'beta' in param_history:
            ax = axes_flat[plot_idx]
            ax.plot(epochs, param_history['beta'], color=colors[plot_idx], marker='.')
            ax.set_ylabel('beta (Coupling)')
            ax.set_title('beta Evolution')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        if 'tau' in param_history:
            ax = axes_flat[plot_idx]
            ax.plot(epochs, param_history['tau'], color=colors[plot_idx], marker='.')
            ax.set_ylabel('tau (Time Const)')
            ax.set_title('tau Evolution')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Plot mean of vector parameters
        vec_params = ['d_mean', 'alpha_cubic_mean', 'beta_cubic_mean', 'gamma_cubic_mean']
        vec_labels = ['d (Offset)', r'$\alpha_{cubic}$', r'$\beta_{cubic}$', r'$\gamma_{cubic}$'] # Use LaTeX for greek letters
        for name, label in zip(vec_params, vec_labels):
            if name in param_history and plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                ax.plot(epochs, param_history[name], color=colors[plot_idx], marker='.')
                ax.set_ylabel(f'Mean {label}')
                ax.set_title(f'Mean {label} Evolution')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        # Plot norm of matrix parameters
        mat_params = ['A_weight_norm', 'proj_weight_norm']
        mat_labels = ['A (Synaptic)', 'Projection']
        for name, label in zip(mat_params, mat_labels):
            if name in param_history and plot_idx < len(axes_flat):
                ax = axes_flat[plot_idx]
                ax.plot(epochs, param_history[name], color=colors[plot_idx], marker='.')
                ax.set_ylabel(f'Frobenius Norm')
                ax.set_title(f'{label} Weight Norm Evolution')
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        # Hide any unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # Common xlabel
        # Add xlabel only to the plots in the last row
        last_row_start_index = (n_rows - 1) * n_cols
        for i in range(last_row_start_index, min(plot_idx, len(axes_flat))):
            axes_flat[i].set_xlabel('Epoch')

        fig.suptitle(f'Trainable Parameter Evolution (Trial {self.sheet_name})', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

        if save_path:
            plt.savefig(save_path)
            print(f"Parameter evolution plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

# -------------------------HELPER FUNCTIONS-------------------------
    def create_neuron_mapping(self, connectome_path):
        """Use the pre-aligned neuron indices from dataset"""
        # We can simplify this since neurons are already aligned with connectome
        
        # Create mapping from dataset indices to simulation indices
        self.dataset_to_sim_idx = {}
        self.sim_to_dataset_idx = self.dataset.sim_to_dataset_idx
        # ADDED: Create the list of sim_indices ordered by dataset_idx (0 to M-1)
        dataset_indices = list(range(self.n_neurons))
        self.sim_indices_for_observed = torch.tensor(list(self.sim_to_dataset_idx.keys()), device=self.device, dtype= torch.long)
        # Create projection mask (initialized with zeros)
        self.proj_mask = torch.zeros(self.latent_dim, self.n_sim_neurons, device=self.device)
        
        # Since neurons are already ordered by connectome index, their position in the array
        # corresponds to their order in the simulation
        for dataset_idx in range(self.n_neurons):
            # Find the corresponding sim_idx for this neuron
            for sim_idx, mapped_dataset_idx in self.sim_to_dataset_idx.items():
                if mapped_dataset_idx == dataset_idx:
                    self.dataset_to_sim_idx[dataset_idx] = sim_idx
                    self.proj_mask[:, sim_idx] = 1.0
                    break
        
        print(f"Found {len(self.dataset_to_sim_idx)} overlapping neurons between dataset and simulation")
        print(f"Dataset has {self.n_neurons} neurons, simulation has {self.n_sim_neurons} neurons")
    
    def initialize_projection(self):
        """Initialize the projection matrix with masked PCA components"""
        # Initialize projection weight with zeros
        self.projection.weight.data.zero_()
        
        # For each neuron in our dataset that overlaps with the simulation
        for dataset_idx, sim_idx in self.dataset_to_sim_idx.items():
            # For each latent dimension
            for dim in range(self.latent_dim):
                # Set the weight to the PCA component value
                self.projection.weight.data[dim, sim_idx] = torch.tensor(
                    self.dataset.components[dim, dataset_idx],
                    dtype=torch.float32,
                    device=self.device
                )

        # Initialize projection weight with PCA components
        self.observed_projection.weight.data = torch.tensor(
            self.dataset.components, # Shape: [latent_dim, n_neurons]
            dtype=torch.float32,
            device=self.device
        )
        # Initialize bias to zero
        self.observed_projection.bias.data.zero_()
        # print(f"Observed projection matrix (L={self.latent_dim} x M={self.n_neurons}) initialized with PCA weights.")

        # Apply the mask to ensure only overlapping neurons are used
        self.projection.weight.data *= self.proj_mask
        
        # Initialize bias to zero
        self.projection.bias.data.zero_()
        
    def _calculate_neural_covariance(self, neural_activity_M_Tsim, weights_1d_Tsim, epsilon=1e-8):

        if self.n_neurons == 0: # No observed neurons
            return np.zeros((0,0))

        valid_time_indices = torch.where(weights_1d_Tsim > 0)[0]

        if len(valid_time_indices) < 2:
            print("Warning: Less than 2 valid time points for covariance calculation. Returning zero matrix.")
            # Ensure the returned zero matrix matches the number of neurons M from neural_activity_M_Tsim
            return np.zeros((neural_activity_M_Tsim.shape[0], neural_activity_M_Tsim.shape[0]))

        # Select activity only at the actual frames
        if len(weights_1d_Tsim) == neural_activity_M_Tsim.shape[1]:
            activity_at_frames_M_Tactual = neural_activity_M_Tsim[:, valid_time_indices]  # [M, T_actual]
        else:
            activity_at_frames_M_Tactual = neural_activity_M_Tsim

        # 1. Center the data (already in your code)
        mean_activity = torch.mean(activity_at_frames_M_Tactual, dim=1, keepdim=True)
        centered_activity = activity_at_frames_M_Tactual - mean_activity

        # 2. Standardize the data (New part)
        # Calculate standard deviation for each neuron across the valid time points
        # ddof=0 for population std, ddof=1 for sample std. For consistency with N-1 later,
        # using N (ddof=0) for std here and then dividing by (N-1) for covariance is fine.
        # Or, calculate std with N-1 (unbiased) and also use N-1 in covariance.
        # Let's use torch.std which by default uses N, and adjust later if strict sample covariance is needed.
        # However, since we scale by it, for correlation, the exact scaling factor in std cancels out
        # as long as it's consistent. Let's use N-1 for std to be conventional.
        n_actual_frames = activity_at_frames_M_Tactual.shape[1]
        if n_actual_frames <= 1: # Need at least 2 points to calculate std dev meaningfully for N-1
            print("Warning: Not enough actual frames for std dev calculation (<=1). Returning zero matrix.")
            print(f"the shape of activity_at_frames_M_Tactual is {activity_at_frames_M_Tactual.shape}")
            print(f"the shape of weights_1d_Tsim is {weights_1d_Tsim.shape}")
            print(f"valid time points {len(valid_time_indices)}")
            return np.zeros((neural_activity_M_Tsim.shape[0], neural_activity_M_Tsim.shape[0]))

        std_activity = torch.std(activity_at_frames_M_Tactual, dim=1, keepdim=True, unbiased=True) # unbiased=True uses N-1

        # Add epsilon to prevent division by zero for traces with no variance
        standardized_activity = centered_activity / (std_activity + epsilon)
        # Note: if a trace had 0 std, it means it was constant. After centering, it's all zeros.
        # Dividing by epsilon will still result in zeros, which is fine.

        # 3. Calculate covariance: (X_standardized @ X_standardized.T) / (N_actual_frames - 1)
        # This will now be the correlation matrix because the data was standardized.
        # The (N_actual_frames - 1) is the standard unbiased estimator for covariance.
        # For correlation, it's also often shown as C = (1/N) * Z * Z.T where Z are z-scored variables.
        # Or C = (1/(N-1)) * Z * Z.T. Both are valid definitions of correlation matrix from standardized vars.
        # Let's stick to (N-1) for consistency with typical covariance definition.

        # The n_actual_frames check is already above, so we can proceed.
        covariance_matrix = torch.matmul(standardized_activity, standardized_activity.t()) / (n_actual_frames - 1)
        
        # If you strictly want the correlation matrix where diagonal elements are exactly 1 (except for edge cases):
        # After standardization, the covariance IS the correlation.
        # The (n_actual_frames - 1) factor is typical for sample covariance.
        # If using `np.corrcoef`, it would handle this internally.
        # A common definition for sample correlation matrix is just (1/(N-1)) * Z Z^T where Z is data standardized using sample std dev.
        # So your current calculation with standardized_activity will yield the correlation matrix.

        return covariance_matrix.cpu().detach().numpy()

    def plot_covariance_heatmaps(self, actual_cov, sim_cov_before, sim_cov_after, save_path=None):
        """
        Plots three neural covariance matrices as heatmaps in a single figure.
        Args:
            actual_cov (np.ndarray): The actual neural covariance matrix.
            sim_cov_before (np.ndarray): Simulated neural covariance matrix before training.
            sim_cov_after (np.ndarray): Simulated neural covariance matrix after training.
            save_path (str, optional): Path to save the figure.
        """
        if actual_cov is None or sim_cov_before is None or sim_cov_after is None :
            print("Warning: One or more covariance matrices are None. Skipping heatmap plotting.")
            if actual_cov is not None and actual_cov.shape == (0,0): # Special case for no neurons
                 print("No observed neurons to plot covariance for.")
            return
        if actual_cov.shape[0] == 0: # No observed neurons
            print("No observed neurons to plot covariance for.")
            return


        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmins = []
        vmaxs = []
        for cov_matrix in [actual_cov, sim_cov_before, sim_cov_after]:
            if cov_matrix is not None and cov_matrix.size > 0:
                 #Ensure finite values for min/max calculation
                finite_vals = cov_matrix[np.isfinite(cov_matrix)]
                if finite_vals.size > 0:
                    vmins.append(np.min(finite_vals))
                    vmaxs.append(np.max(finite_vals))
        
        # Use global min/max for consistent color scaling if there are valid values
        # Otherwise, default to a sensible range or let imshow auto-scale
        vmin = min(vmins) if vmins else -1 
        vmax = max(vmaxs) if vmaxs else 1
        if vmin == vmax : # if all values are the same
            vmin -= 0.5
            vmax += 0.5
        if not (np.isfinite(vmin) and np.isfinite(vmax)): # Fallback if min/max are not finite
            vmin, vmax = -1,1


        titles = ['Actual Neural Covariance', 'Simulated Cov (Before Training)', 'Simulated Cov (After Training)']
        matrices = [actual_cov, sim_cov_before, sim_cov_after]

        for i, ax in enumerate(axes):
            if matrices[i] is not None and matrices[i].size > 0:
                im = ax.imshow(matrices[i], aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, orientation='vertical')
                ax.set_title(titles[i])
                ax.set_xlabel("Neuron Index")
                ax.set_ylabel("Neuron Index")
            else:
                ax.set_title(f"{titles[i]}\n(Not Available)")
                ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Covariance heatmaps saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

class FishSpikeDataset(torch.utils.data.Dataset):
    """
    Dataset for loading fish spike data for covariance-based model fitting.
    Uses load_fish_spike_data from load.py.
    """
    def __init__(self, fish_id='201106', sequence_length=None, device='cpu', n_neurons = None):
        super().__init__()
        self.fish_id = fish_id
        self.device = device
        
        # Load data using the function from load.py
        # load_fish_spike_data returns a numpy array (neurons, timepoints)
        self.raw_spike_data_np = load_fish_spike_data(ifish=self.fish_id, n_neurons=n_neurons)
        self.n_neurons, self.total_timepoints = self.raw_spike_data_np.shape

        if sequence_length is None:
            self.sequence_length = self.total_timepoints
        else:
            self.sequence_length = min(sequence_length, self.total_timepoints)

        # Use the first 'sequence_length' timepoints
        self.spike_data_np = self.raw_spike_data_np[:, :self.sequence_length]
        self.spike_data = torch.tensor(self.spike_data_np, dtype=torch.float32, device=self.device)

        self.x0 = torch.zeros(1, self.n_neurons, dtype=torch.float32, device=device)
        for i in range(self.n_neurons):
            self.x0[0, i] = self.spike_data[i, 0]

    def __len__(self):
        # This dataset handles a single sequence chunk.
        return 1

    def __getitem__(self, idx):
        # Returns the spike data chunk: (neurons, timepoints)
        if idx >= 1:
            raise IndexError("This dataset currently only supports a single sequence chunk.")
        return self.spike_data

    def get_full_data(self):
        """Helper to get the processed spike data tensor."""
        return self.spike_data

class SpikeCovarianceModel(nn.Module):
    """
    A minimal model that wraps SpikeSimulator and trains its parameters
    by fitting the covariance matrix of simulated spike activity to a target
    covariance matrix derived from fish spike data.
    """
    def __init__(self, fish_id='201106', sequence_length=1000,
                 beta_init=1.0, tau_init=5.0, noise_strength=0.1, weight_scale=1.0,
                 learning_rate=0.01, device=None, n_neurons = None):
        super().__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = device
        
        print(f"SpikeCovarianceModel using device: {self.device}")

        self.fish_id = fish_id
        # sequence_length determines the number of time points for simulation and target data
        self.sequence_length = sequence_length 

        # Initialize dataset
        self.dataset = FishSpikeDataset(fish_id=self.fish_id,
                                        sequence_length=self.sequence_length,
                                        device=self.device,
                                        n_neurons=n_neurons)
        
        self.n_neurons = self.dataset.n_neurons

        self.simulator = SpikeSimulator(fish_id=self.fish_id,
                                        beta=beta_init,
                                        tau=tau_init,
                                        noise_strength=noise_strength,
                                        weight_scale=weight_scale,
                                        n_neurons=n_neurons).to(self.device)
        
        # Validate neuron count consistency
        if self.simulator.n_units != self.n_neurons:
            raise ValueError(
                f"Mismatch in neuron count: Dataset has {self.n_neurons}, "
                f"Simulator (from fish_id {self.fish_id}) initialized with {self.simulator.n_units}."
            )

        # Get target spike data and pre-calculate target covariance matrix
        self.target_spikes_N_T = self.dataset.get_full_data() # Shape: (n_neurons, sequence_length)
        self.target_cov = self._calculate_covariance(self.target_spikes_N_T)
        self.target_corr = self._calculate_correlation(self.target_spikes_N_T)

        # Optimizer for the simulator's parameters
        self.optimizer = torch.optim.Adam(self.simulator.parameters(), lr=learning_rate)

    def forward(self, x0_N=None, t_span=None, fps = 10.0):
        """
        Runs the simulation using the SpikeSimulator.

        Args:
            x0_N (Tensor, optional): Initial condition for neurons. Shape (n_neurons,).
                                     If None, simulator's default is used.
            t_span (tuple, optional): (start_time, end_time) for simulation.
            time_steps (int, optional): Number of simulation steps.

        Returns:
            Tensor: Simulated spike states from the simulator. Shape (time_steps, batch_size, n_neurons).
        """
        if x0_N is not None:
            # Add batch dimension for simulator: (1, n_neurons)
            x0_sim = x0_N.unsqueeze(0).to(self.device) 
        else:
            x0_sim = None # SpikeSimulator handles default x0 if None

        time_steps = self.sequence_length

        t_start = 0.0
        t_end = time_steps / fps
        t_span = (t_start, t_end) 
        
        # SpikeSimulator.simulate returns t, states
        # states shape: (time_steps, batch_size, n_units)
        _t, simulated_spikes_T_B_N = self.simulator.simulate(t_span=t_span,
                                                             x0=x0_sim,
                                                             time_steps=time_steps)
        return simulated_spikes_T_B_N

    def fit(self, save_path_prefix = None, n_epochs=100, x0_N=None):
        """
        Trains the model by fitting the simulated covariance to the target covariance.

        Args:
            n_epochs (int): Number of training epochs.
            x0_N (Tensor, optional): Initial condition for simulations. Shape (n_neurons,).
                                     If None, uses the first time point of target data.
        Returns:
            dict: Training history containing loss per epoch.
        """
        if x0_N is None:
            initial_condition_N = self.target_spikes_N_T[:, 0].clone().detach()
        else:
            initial_condition_N = x0_N.to(self.device) # Ensure it's on the correct device

        self.simulator.eval() 
        with torch.no_grad():
            sim_spikes_before = self.forward(x0_N=initial_condition_N)
            sim_spikes_before_N_T = sim_spikes_before.squeeze(1).permute(1, 0)
            sim_corr_before_training = self._calculate_correlation(sim_spikes_before_N_T)

        self.train()
        self.simulator.train()
        training_history = {'loss': []}

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Run simulation
            simulated_spikes_T_B_N = self.forward(x0_N=initial_condition_N)
            
            simulated_spikes_N_T = simulated_spikes_T_B_N.squeeze(1).permute(1, 0)

            # Calculate covariance of simulated data
            simulated_cov = self._calculate_covariance(simulated_spikes_N_T)

            # Loss: MSE between target and simulated covariance matrices
            loss = F.mse_loss(simulated_cov, self.target_cov)

            if torch.isnan(loss):
                print(f"Epoch {epoch+1}: NaN loss detected. Stopping training.")
                break 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.simulator.parameters(), max_norm=1.0)
            self.optimizer.step()

            training_history['loss'].append(loss.item())
            if (epoch + 1) % 1== 0 or epoch == 0 or epoch == n_epochs -1 :
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.12f}")
        
                
        self.simulator.eval() 
        with torch.no_grad():
            sim_spikes_after = self.forward(x0_N=initial_condition_N)
            sim_spikes_after_N_T = sim_spikes_after.squeeze(1).permute(1, 0)
            sim_corr_after_training = self._calculate_correlation(sim_spikes_after_N_T)

        if save_path_prefix:
            save_dir = Path(save_path_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
            loss_plot_path = str(save_dir / "loss_progression.png")
            cov_plot_path = str(save_dir / "covariance_heatmaps.png")
        else:
            loss_plot_path = None
            cov_plot_path = None

        self.plot_loss_progression(training_history['loss'], save_path=loss_plot_path)
        self.plot_covariance_heatmaps(
            actual_cov=self.target_corr, 
            sim_cov_before=sim_corr_before_training,
            sim_cov_after=sim_corr_after_training,
            save_path=cov_plot_path
        )

        return training_history
#---------------------------HELPERS----------------------------
    def _calculate_covariance(self, data_N_T):
        if data_N_T.shape[1] < 2: # torch.cov requires at least 2 observations
            print("Warning: Not enough timepoints (<2) to calculate covariance. Returning zero matrix.")
            return torch.zeros((data_N_T.shape[0], data_N_T.shape[0]), device=self.device, dtype=data_N_T.dtype)
        # torch.cov expects observations as columns, features (neurons) as rows.
        return torch.cov(data_N_T)

    def _calculate_correlation(self, neural_activity_M_Tsim, epsilon=1e-8):
        if neural_activity_M_Tsim.shape[0] == 0: # No observed neurons
            return np.zeros((0,0))

        if neural_activity_M_Tsim.shape[1] < 2: # Need at least 2 timepoints
            print("Warning: Less than 2 time points for covariance calculation. Returning zero matrix.")
            return np.zeros((neural_activity_M_Tsim.shape[0], neural_activity_M_Tsim.shape[0]))

        # All provided timepoints are considered valid
        activity_at_frames_M_Tactual = neural_activity_M_Tsim

        mean_activity = torch.mean(activity_at_frames_M_Tactual, dim=1, keepdim=True)
        centered_activity = activity_at_frames_M_Tactual - mean_activity

        n_actual_frames = activity_at_frames_M_Tactual.shape[1]
        # This check is technically redundant if the one above (neural_activity_M_Tsim.shape[1] < 2) is passed,
        # but kept for safety, though n_actual_frames will be >= 2 here.
        if n_actual_frames <= 1: 
            print("Warning: Not enough actual frames for std dev calculation (<=1). Returning zero matrix.")
            return np.zeros((neural_activity_M_Tsim.shape[0], neural_activity_M_Tsim.shape[0]))

        std_activity = torch.std(activity_at_frames_M_Tactual, dim=1, keepdim=True, unbiased=True) 
        standardized_activity = centered_activity / (std_activity + epsilon)
        
        covariance_matrix = torch.matmul(standardized_activity, standardized_activity.t()) / (n_actual_frames - 1)
        
        return covariance_matrix.cpu().detach().numpy()

    def plot_loss_progression(self, loss_history, save_path=None):
        if not loss_history:
            print("No loss history to plot.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Covariance MSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss Progression (Fish {self.fish_id})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        try:
            plt.yscale('log') 
        except ValueError: pass 
        if save_path:
            plt.savefig(save_path)
            print(f"Loss progression plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_covariance_heatmaps(self, actual_cov, sim_cov_before, sim_cov_after, save_path=None):
            """
            Plots three neural covariance matrices as heatmaps in a single figure.
            Args:
                actual_cov (np.ndarray): The actual neural covariance matrix.
                sim_cov_before (np.ndarray): Simulated neural covariance matrix before training.
                sim_cov_after (np.ndarray): Simulated neural covariance matrix after training.
                save_path (str, optional): Path to save the figure.
            """
            if actual_cov is None or sim_cov_before is None or sim_cov_after is None :
                print("Warning: One or more covariance matrices are None. Skipping heatmap plotting.")
                if actual_cov is not None and actual_cov.shape == (0,0): 
                    print("No observed neurons to plot covariance for.")
                return
            if actual_cov.shape[0] == 0: 
                print("No observed neurons to plot covariance for.")
                return

            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5)) 
            vmins, vmaxs = [], []
            all_matrices = [m for m in [actual_cov, sim_cov_before, sim_cov_after] if m is not None and m.size > 0]

            if not all_matrices:
                print("No valid covariance matrices to plot.")
                plt.close(fig) 
                return

            for cov_matrix in all_matrices:
                finite_vals = cov_matrix[np.isfinite(cov_matrix)]
                if finite_vals.size > 0:
                    vmins.append(np.min(finite_vals))
                    vmaxs.append(np.max(finite_vals))
            
            vmin = min(vmins) if vmins else -1 
            vmax = max(vmaxs) if vmaxs else 1
            if vmin == vmax : 
                vmin -= 0.5
                vmax += 0.5
            if not (np.isfinite(vmin) and np.isfinite(vmax)):
                vmin, vmax = -1, 1

            titles = [
                f'Target Covariance (Fish {self.fish_id})', 
                'Simulated Cov (Before Training)',
                'Simulated Cov (After Training)'
            ]
            matrices_to_plot = [actual_cov, sim_cov_before, sim_cov_after]

            for i, ax in enumerate(axes):
                if matrices_to_plot[i] is not None and matrices_to_plot[i].size > 0:
                    im = ax.imshow(matrices_to_plot[i], aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax) 
                    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04) 
                    ax.set_title(titles[i], fontsize=10) 
                    ax.set_xlabel("Neuron Index", fontsize=8) 
                    if i == 0:
                        ax.set_ylabel("Neuron Index", fontsize=8) 
                    ax.tick_params(axis='both', which='major', labelsize=7) 
                else:
                    ax.set_title(f"{titles[i]}\n(Not Available)", fontsize=10)
                    ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            fig.suptitle(f'Covariance Matrix Comparison (Fish {self.fish_id})', fontsize=12, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.95]) 

            if save_path:
                plt.savefig(save_path, dpi=150) 
                print(f"Covariance heatmaps saved to {save_path}")
                plt.close(fig)
            else:
                plt.show()

# -----------------------------HELPER METHODS----------------------------    
def load_identified_data(data_dir='data/activity/WT_NoStim', sheet_name=0, 
                        smooth=True, sigma=10.0, detrend=True,
                        connectome_path='data/connectome/White1986'):
    """
    Load neural activity data, keeping only neurons that overlap with the connectome,
    optionally smoothing and detrending the traces.
    Permutes the order of neurons to match the connectome ordering.
    
    Parameters:
    - data_dir: Directory containing IDs.xlsx, traces.xlsx, and tracesDif.xlsx
    - sheet_name: Which sheet to read from Excel files
    - smooth: Whether to apply Gaussian smoothing (default: True) 
    - sigma: Standard deviation of the Gaussian kernel for smoothing (default: 15.0)
    - detrend: Whether to subtract a linear trend from each neuron's trace (default: True)
    - connectome_path: Path to the connectome data
    
    Returns:
    - identified_traces: Tensor of shape [n_identified_neurons, n_timepoints] ordered and processed
    - identified_traces_dif: Tensor of shape [n_identified_neurons, n_timepoints] ordered (not smoothed/detrended)
    - neuron_ids: Dictionary mapping neuron names to indices (in connectome ordering)
    - fps: Frames per second
    - sim_to_dataset_idx: Dictionary mapping simulation indices to dataset indices
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load connectome data first
    _, connectome_neuron_ids = load_synaptic_data()
    
    # Load IDs with specified sheet
    ids_path = Path(data_dir) / 'IDs.xlsx'
    if not ids_path.exists():
        raise FileNotFoundError(f"IDs.xlsx not found in {data_dir}")
    
    try:
        ids_df = pd.read_excel(ids_path, header=None, sheet_name=sheet_name)
        
        # Filter to only include proper neuron names
        original_neuron_ids = {}
        for idx, name in enumerate(ids_df.iloc[0]):
            if pd.notna(name):
                name_str = str(name).strip("'").strip()
                if (name_str and name_str != '[]' and 
                    not name_str.isdigit() and
                    any(c.isalpha() for c in name_str)):
                    original_neuron_ids[name_str] = idx

    except Exception as e:
        raise Exception(f"Error reading IDs.xlsx: {e}")

    # Find overlapping neurons
    overlapping_neurons = []
    for name, orig_idx in original_neuron_ids.items():
        if name in connectome_neuron_ids:
            sim_idx = connectome_neuron_ids[name]
            overlapping_neurons.append((name, orig_idx, sim_idx))
    
    if not overlapping_neurons:
         raise ValueError("No overlapping neurons found. Cannot proceed.")

    # Sort by simulation index
    overlapping_neurons.sort(key=lambda x: x[2])
    
    # Load traces
    traces_path = Path(data_dir) / 'traces.xlsx'
    traces_dif_path = Path(data_dir) / 'tracesDif.xlsx'

    if not traces_path.exists():
        raise FileNotFoundError(f"traces.xlsx not found in {data_dir}")
    
    try:
        traces_df = pd.read_excel(traces_path, sheet_name=sheet_name)
        traces_raw = torch.tensor(traces_df.values, dtype=torch.float32, device=device)
        
        traces_dif_df = pd.read_excel(traces_dif_path, sheet_name=sheet_name)
        traces_dif_raw = torch.tensor(traces_dif_df.values, dtype=torch.float32, device=device)

        # Standardize both datasets
        traces_mean = traces_raw.mean(dim=0, keepdim=True)
        traces_std = traces_raw.std(dim=0, keepdim=True)
        traces_standardized = (traces_raw - traces_mean) / (traces_std + 1e-6) # Add epsilon for stability

        traces_dif_mean = traces_dif_raw.mean(dim=0, keepdim=True)
        traces_dif_std = traces_dif_raw.std(dim=0, keepdim=True)
        traces_dif_standardized = (traces_dif_raw - traces_dif_mean) / (traces_dif_std + 1e-6)
        
        # Transpose to (neurons, timepoints)
        traces_t = traces_standardized.cpu().numpy().T
        traces_dif_t = traces_dif_standardized.cpu().numpy().T

        traces = torch.tensor(traces_t, dtype=torch.float32, device=device)
        traces_dif = torch.tensor(traces_dif_t, dtype=torch.float32, device=device)
        
        # --- Processing Steps ---
        # 1. Select overlapping neurons in connectome order
        original_indices = [tup[1] for tup in overlapping_neurons]
        identified_traces = traces[original_indices]
        identified_traces_dif = traces_dif[original_indices] # Keep original diff traces separate

        # 2. Smooth (if requested) - applied only to main traces
        if smooth:
            identified_traces = gaussian_smooth(identified_traces, sigma)

        # 3. Detrend (if requested) - applied only to main traces
        if detrend:
            traces_np = identified_traces.cpu().numpy()
            n_neurons, n_timepoints = traces_np.shape
            time = np.arange(n_timepoints)
            detrended_traces_np = np.zeros_like(traces_np)

            for i in range(n_neurons):
                try:
                    coeffs = np.polyfit(time, traces_np[i, :], 1)
                    trend = np.polyval(coeffs, time)
                    detrended_traces_np[i, :] = traces_np[i, :] - trend
                except (np.linalg.LinAlgError, ValueError) as fit_err:
                    print(f"  Warning: Could not detrend neuron index {i}. Using original trace. Error: {fit_err}")
                    detrended_traces_np[i, :] = traces_np[i, :] # Keep original if detrend fails

            identified_traces = torch.tensor(detrended_traces_np, dtype=torch.float32, device=device)

        # --- Create final mappings ---
        neuron_ids = {}
        sim_to_dataset_idx = {}
        for new_idx, (name, _, sim_idx) in enumerate(overlapping_neurons):
            neuron_ids[name] = new_idx
            sim_to_dataset_idx[sim_idx] = new_idx
        

    except Exception as e:
        raise Exception(f"Error processing traces: {e}")

    # Load fps
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
        
    # Note: identified_traces_dif is returned without smoothing or detrending
    return identified_traces, identified_traces_dif, neuron_ids, fps, sim_to_dataset_idx

# Define helper functions for new losses
def weighted_covariance(tensor1, tensor2, weights):
    """Calculates weighted covariance matrices and the Frobenius norm of their difference.
       Assumes inputs are [L, T] and weights are [L, T] or [T].
    """
    L, T = tensor1.shape
    if weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(L, T) # Broadcast weights across latent dims

    # Mask out invalid time points (where weight is 0)
    # We need to compute mean only over valid points
    valid_mask = weights > 0
    num_valid_points = valid_mask.sum(dim=1, keepdim=True).float()

    # Avoid division by zero if a dimension has no valid points
    num_valid_points = torch.clamp(num_valid_points, min=1.0)

    # Calculate weighted mean for valid points
    mean1 = torch.sum(tensor1 * weights, dim=1, keepdim=True) / num_valid_points
    mean2 = torch.sum(tensor2 * weights, dim=1, keepdim=True) / num_valid_points

    # Center data using weighted mean, only considering valid points
    centered1 = (tensor1 - mean1) * valid_mask
    centered2 = (tensor2 - mean2) * valid_mask

    # Calculate weighted covariance matrix
    # Use N-1 normalization where N is the number of valid points
    # Need to be careful if num_valid_points varies per dimension. Use average? Or minimum?
    # Let's use the minimum number of valid points across dimensions for normalization factor
    N_eff = torch.min(num_valid_points).item()
    norm_factor = max(1.0, N_eff - 1.0) # Ensure at least 1

    cov1 = torch.matmul(centered1, centered1.t()) / norm_factor
    cov2 = torch.matmul(centered2, centered2.t()) / norm_factor

    # Calculate Frobenius norm of the difference
    cov_diff = torch.sqrt(torch.sum((cov1 - cov2)**2))
    return cov_diff

def weighted_pearson_correlation(tensor1, tensor2, weights):
    """Calculates weighted Pearson correlation for each latent dimension.
       Assumes inputs are [L, T] and weights are [L, T] or [T].
       Returns the average (1 - correlation) across dimensions.
    """
    L, T = tensor1.shape
    if weights.dim() == 1:
        weights = weights.unsqueeze(0).expand(L, T)

    valid_mask = weights > 0
    num_valid_points = valid_mask.sum(dim=1).float()
    num_valid_points = torch.clamp(num_valid_points, min=2.0) # Need at least 2 points for correlation

    # Weighted mean
    mean1 = torch.sum(tensor1 * weights, dim=1, keepdim=True) / num_valid_points.unsqueeze(1)
    mean2 = torch.sum(tensor2 * weights, dim=1, keepdim=True) / num_valid_points.unsqueeze(1)

    # Weighted centered values
    centered1 = (tensor1 - mean1) * valid_mask
    centered2 = (tensor2 - mean2) * valid_mask

    # Weighted covariance for numerator
    cov_num = torch.sum(centered1 * centered2 * weights, dim=1)

    # Weighted standard deviation for denominator
    std1 = torch.sqrt(torch.sum(centered1**2 * weights, dim=1) / num_valid_points)
    std2 = torch.sqrt(torch.sum(centered2**2 * weights, dim=1) / num_valid_points)

    # Pearson correlation coefficient per dimension
    # Add small epsilon to prevent division by zero
    corr = cov_num / (std1 * std2 + 1e-6)

    # Clamp correlation between -1 and 1 due to potential numerical issues
    corr = torch.clamp(corr, -1.0, 1.0)

    # Average (1 - correlation) across dimensions where correlation is valid
    valid_dims = num_valid_points >= 2
    if valid_dims.sum() > 0:
        avg_neg_corr = torch.mean(1.0 - corr[valid_dims])
    else:
        avg_neg_corr = torch.tensor(1.0, device=tensor1.device) # Default loss if no valid dims

    return avg_neg_corr


def run_single_training(constrained, train_gap_junctions, sheet_name, latent_dim, sequence_length, test_seq_length, detrend, zoomin_factor, n_epochs, learning_rate, weight_decay, lambda_cov, lambda_corr, scheduler_patience, scheduler_factor, checkpoint_frequency, base_checkpoint_path, base_fig_path, common_params):
    """Runs training for one configuration (constrained or not)."""
    print(f"\n--- Running Training (Constrained={constrained}) ---")

    # Create specific checkpoint path for this run
    checkpoint_path = f"{base_checkpoint_path}_constrained:{constrained}"
    figs_path = f"{base_fig_path}_constrained:{constrained}"

    # Create the model instance
    try:
         pca_model = TrialPCA(
             latent_dim=latent_dim,
             beta=common_params['beta'],
             tau=common_params['tau'],
             zoomin_factor=zoomin_factor,
             sheet_name=sheet_name,
             data_dir=common_params['data_dir'],
             connectome_path=common_params['connectome_path'],
             sequence_length=sequence_length,
             test_seq_length=test_seq_length,
             detrend=detrend,
             noise_strength=common_params['noise_strength'],
             synaptic_gain=common_params['synaptic_gain'],
             constrained=constrained,
             train_gap_junctions = train_gap_junctions
         )
    except Exception as e:
         print(f"Error creating model (Constrained={constrained}): {e}")
         return None # Indicate failure

    # Visualize before training
    pca_model.visualize_prediction(sequence_length=sequence_length, save_path=f"{figs_path}_train_before.png")
    pca_model.visualize_prediction_test(test_sequence_length=test_seq_length, save_path=f"{figs_path}_test_before.png")

    # Train the model
    try:
        pca_model.fit(
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lambda_cov=lambda_cov,
            lambda_corr=lambda_corr,
            scheduler_patience=scheduler_patience,
            scheduler_factor=scheduler_factor,
            checkpoint_path=checkpoint_path, # Use specific path
            checkpoint_frequency=checkpoint_frequency,
            resume_from=None # Always start fresh for comparison run
        )
    except Exception as e:
        print(f"Error during training (Constrained={constrained}): {e}")
        return None # Indicate failure

    # Visualize after training
    pca_model.visualize_prediction(sequence_length=sequence_length, save_path=f"{figs_path}_train_after.png")
    pca_model.visualize_prediction_test(test_sequence_length=test_seq_length, save_path=f"{figs_path}_test_after.png")

    print(f"--- Training Finished (Constrained={constrained}) ---")
    # Return the path to the final checkpoint
    return f"{checkpoint_path}_final.pt"


def plot_comparison_losses(ckpt_path_true, ckpt_path_false, save_path):
    """Loads loss histories from two checkpoints and plots comparisons."""
    print(f"Loading losses from True: {ckpt_path_true}")
    print(f"Loading losses from False: {ckpt_path_false}")

    try:
        map_loc = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        ckpt_true = torch.load(ckpt_path_true, map_location=map_loc)
        ckpt_false = torch.load(ckpt_path_false, map_location=map_loc)
    except Exception as e:
        print(f"Error loading checkpoints for comparison plot: {e}")
        return

    histories = {
        'True': ckpt_true,
        'False': ckpt_false
    }

    metrics_to_plot = [
        ('train_losses_mse', 'Train Latent MSE'),
        ('test_losses_mse', 'Test Latent MSE'), # Use renamed key
        ('train_losses_neural_mse', 'Train Neural MSE'),
        ('train_cov_similarities', 'Train Neural Cov Sim'),
        ('test_cov_similarities', 'Test Neural Cov Sim'),
    ]

    num_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes] # Make iterable

    colors = {'True': 'blue', 'False': 'red'}
    styles = {'True': '-', 'False': '--'}

    max_epochs = 0
    for key, label in metrics_to_plot:
        if key in histories['True']:
             max_epochs = max(max_epochs, len(histories['True'][key]))
        if key in histories['False']:
             max_epochs = max(max_epochs, len(histories['False'][key]))

    if max_epochs == 0:
        print("No history data found in checkpoints.")
        return

    epochs = range(1, max_epochs + 1)

    for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[i]
        for constrained_flag in ['True', 'False']:
            history = histories[constrained_flag].get(metric_key, [])
             # Pad history with NaNs if shorter than max_epochs
            if len(history) < max_epochs:
                 history.extend([np.nan] * (max_epochs - len(history)))
            elif len(history) > max_epochs: # Should not happen if loaded from same run config
                 history = history[:max_epochs]

            # Filter NaNs for plotting
            valid_indices = [e_idx for e_idx, val in enumerate(history) if not np.isnan(val)]
            if valid_indices:
                 valid_epochs = [epochs[vi] for vi in valid_indices]
                 valid_values = [history[vi] for vi in valid_indices]
                 ax.plot(valid_epochs, valid_values,
                         color=colors[constrained_flag],
                         linestyle=styles[constrained_flag],
                         alpha=0.8, label=f'{metric_label} (Constrained={constrained_flag})')

        # Formatting based on metric type
        if 'MSE' in metric_label:
            try: ax.set_yscale('log')
            except ValueError: pass # Keep linear if log fails
            ax.set_ylabel('MSE Loss')
        elif 'Sim' in metric_label:
            ax.set_ylim(-0.75, 1.0)
            ax.set_ylabel('Cosine Similarity')
        else:
            ax.set_ylabel(metric_label) # Default

        ax.set_title(f'{metric_label} Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Epoch')
    fig.suptitle('Constrained vs Unconstrained Training Comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Comparison loss plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_pc_comparison_html(ckpt_path_true, ckpt_path_false, dataset_for_sim, sequence_length, common_params, save_path):
    """Generates an interactive 3D plot comparing latent PCs."""
    print("Generating 3D PC comparison plot...")

    try:
        map_loc = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        ckpt_true = torch.load(ckpt_path_true, map_location=map_loc)
        ckpt_false = torch.load(ckpt_path_false, map_location=map_loc)
    except Exception as e:
        print(f"Error loading checkpoints for HTML plot: {e}")
        return

    models = {}
    latent_states_sim = {}

    # Instantiate and load models
    for flag, ckpt in [('True', ckpt_true), ('False', ckpt_false)]:
         try:
             model = TrialPCA(
                 latent_dim=ckpt.get('latent_dim', 5), # Get latent dim from checkpoint or default
                 beta=common_params['beta'], tau=common_params['tau'], # Use common base params
                 zoomin_factor=common_params['zoomin_factor'], sheet_name=common_params['sheet_name'],
                 data_dir=common_params['data_dir'], connectome_path=common_params['connectome_path'],
                 sequence_length=sequence_length, test_seq_length=0, # Not needed for this sim
                 detrend=common_params['detrend'], noise_strength=0, # Run deterministic sim for viz
                 synaptic_gain=common_params['synaptic_gain'], constrained=(flag=='True')
             )
             model.dynamics_model.load_state_dict(ckpt['dynamics_model_state'])
             if 'observed_projection_state' in ckpt:
                 model.observed_projection.load_state_dict(ckpt['observed_projection_state'])
             elif 'projection_state' in ckpt:
                 model.observed_projection.load_state_dict(ckpt['projection_state'])

             model.to(model.device)
             model.eval()
             models[flag] = model
         except Exception as e:
             print(f"Error instantiating or loading model (Constrained={flag}): {e}")
             return # Cannot proceed without both models

    # --- Run Simulation for Visualization ---
    # Get initial condition (same for both)
    x0_sim = torch.zeros(1, models['True'].n_sim_neurons, dtype=torch.float32, device=models['True'].device)
    if models['True'].n_neurons > 0 :
         x0_dataset = dataset_for_sim.x0.to(models['True'].device)
         dataset_to_sim_map = {v: k for k, v in models['True'].sim_to_dataset_idx.items()}
         for dataset_idx in range(models['True'].n_neurons):
             if dataset_idx < x0_dataset.shape[1] and dataset_idx in dataset_to_sim_map:
                 sim_idx = dataset_to_sim_map[dataset_idx]
                 if sim_idx < x0_sim.shape[1]:
                     x0_sim[0, sim_idx] = x0_dataset[0, dataset_idx]

    t_span = (0, sequence_length / dataset_for_sim.fps if dataset_for_sim.fps > 0 else 1.0)
    sim_steps = sequence_length * common_params['zoomin_factor']

    with torch.no_grad():
        for flag, model in models.items():
            try:
                _, _, _, latent_states = model.forward(
                    x0=x0_sim, t_span=t_span, time_steps=sim_steps
                )
                latent_states_sim[flag] = latent_states.squeeze(1).cpu().numpy() # [T_sim, L]
            except Exception as e:
                print(f"Error running simulation for model (Constrained={flag}): {e}")
                latent_states_sim[flag] = None


    # --- Prepare Plot Data ---
    actual_latent = dataset_for_sim.latent_traces[:, :sequence_length].cpu().numpy() # [L, T_actual]
    time_actual = np.linspace(0, t_span[1], actual_latent.shape[1])

    if actual_latent.shape[0] < 3:
         print("Error: Latent dimension < 3. Cannot generate 3D plot.")
         return

    fig = go.Figure()

    # Plot Actual Data
    fig.add_trace(go.Scatter3d(
        x=actual_latent[0, :], y=actual_latent[1, :], z=actual_latent[2, :],
        mode='lines', line=dict(color='red', width=3), name='Actual Data (PCA)',
        customdata=time_actual, hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Time: %{customdata:.2f}s<extra></extra>'
    ))

    # Plot Simulated Data
    colors = {'True': 'blue', 'False': 'green'}
    for flag, latent_sim in latent_states_sim.items():
         if latent_sim is not None and latent_sim.shape[1] >= 3:
             t_sim = np.linspace(0, t_span[1], latent_sim.shape[0])
             fig.add_trace(go.Scatter3d(
                 x=latent_sim[:, 0], y=latent_sim[:, 1], z=latent_sim[:, 2],
                 mode='lines', line=dict(color=colors[flag], width=3), name=f'Simulated (Constrained={flag})',
                 customdata=t_sim, hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>Time: %{customdata:.2f}s<extra></extra>'
             ))

    # Update Layout
    fig.update_layout(
        title=f'Latent Space Comparison (Trial {common_params["sheet_name"]})',
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3',
            aspectmode='cube'
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        legend_title_text='Trace Type'
    )

    # Save HTML
    try:
        pio.write_html(fig, file=save_path, auto_open=False)
        print(f"PC comparison HTML plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving HTML plot: {e}")


def compare_constrain(
    sheet_name = 0,
    latent_dim = 5,
    sequence_length = 300,
    test_seq_length = 150,
    detrend = True,
    zoomin_factor = 5,
    n_epochs = 100,
    learning_rate = 2e-2,
    weight_decay = 1e-4,
    lambda_cov = 1e-3,
    lambda_corr = 1e-3,
    scheduler_patience = 10,
    scheduler_factor = 0.9,
    checkpoint_frequency = 10,
    # Shared initial params for LinearDynamics
    beta_init = 0.02,
    tau_init = 2.0,
    noise_init = 0.01,
    synaptic_gain_init = 7.5,
    # Paths and other common args
    data_dir = 'data/activity/WT_NoStim',
    connectome_path = 'data/connectome/White1986',
    base_checkpoint_path = None, # Base path, flags will be added
    base_fig_path = None, # Base path for figures
    train_gap_junctions = False
    ):
    """
    Trains and compares two models: one with constraints, one without.
    Generates comparison plots for losses and latent space trajectories.
    """
    print(f"\n===== Starting Constrained vs Unconstrained Comparison for Trial {sheet_name} =====")

    # Define base paths if not provided
    if base_checkpoint_path is None:
        base_checkpoint_path = f"models/trial_{sheet_name}_compare"
    if base_fig_path is None:
        base_fig_path = f"figs/trial_{sheet_name}_compare"

    # Create output directories
    Path(base_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(base_fig_path).parent.mkdir(parents=True, exist_ok=True)

    common_params = {
        'beta': beta_init, 'tau': tau_init, 'noise_strength': noise_init,
        'synaptic_gain': synaptic_gain_init, 'zoomin_factor': zoomin_factor,
        'sheet_name': sheet_name, 'data_dir': data_dir,
        'connectome_path': connectome_path, 'detrend': detrend
    }

    # Run training for both conditions
    ckpt_true = run_single_training(
        constrained=True, train_gap_junctions=train_gap_junctions, sheet_name=sheet_name, latent_dim=latent_dim,
        sequence_length=sequence_length, test_seq_length=test_seq_length,
        detrend=detrend, zoomin_factor=zoomin_factor, n_epochs=n_epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        lambda_cov=lambda_cov, lambda_corr=lambda_corr,
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
        checkpoint_frequency=checkpoint_frequency,
        base_checkpoint_path=base_checkpoint_path, base_fig_path=base_fig_path,
        common_params=common_params
    )

    ckpt_false = run_single_training(
        constrained=False, train_gap_junctions=train_gap_junctions, sheet_name=sheet_name, latent_dim=latent_dim,
        sequence_length=sequence_length, test_seq_length=test_seq_length,
        detrend=detrend, zoomin_factor=zoomin_factor, n_epochs=n_epochs,
        learning_rate=learning_rate, weight_decay=weight_decay,
        lambda_cov=lambda_cov, lambda_corr=lambda_corr,
        scheduler_patience=scheduler_patience, scheduler_factor=scheduler_factor,
        checkpoint_frequency=checkpoint_frequency,
        base_checkpoint_path=base_checkpoint_path, base_fig_path=base_fig_path,
        common_params=common_params
    )

    # Generate comparison plots if both runs succeeded
    if ckpt_true and ckpt_false:
        print("\n--- Generating Comparison Plots ---")
        # Plot loss comparison
        plot_comparison_losses(
            ckpt_path_true=ckpt_true,
            ckpt_path_false=ckpt_false,
            save_path=f"{base_fig_path}_loss_comparison.png"
        )

        # Plot PC comparison (HTML)
        # Need a dataset instance to pass for actual data plot
        try:
             dataset_for_plot = IdentifiedPCA(
                 latent_dim=latent_dim, zoomin_factor=zoomin_factor,
                 sequence_length=sequence_length, test_seq_length=test_seq_length,
                 sheet_name=sheet_name, data_dir=data_dir,
                 connectome_path=connectome_path, detrend=detrend
             )
             plot_pc_comparison_html(
                 ckpt_path_true=ckpt_true,
                 ckpt_path_false=ckpt_false,
                 dataset_for_sim=dataset_for_plot, # Pass dataset instance
                 sequence_length=sequence_length,
                 common_params=common_params, # Pass params for model re-instantiation
                 save_path=f"{base_fig_path}_pc_comparison.html"
             )
        except Exception as e:
             print(f"Error generating PC comparison plot: {e}")

    else:
        print("\nSkipping comparison plots as one or both training runs failed.")

    print(f"===== Comparison Finished for Trial {sheet_name} =====")

