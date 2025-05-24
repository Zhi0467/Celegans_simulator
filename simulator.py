"""
The ODESimulator class is the base class for all ODE simulators.
It contains the methods for simulating the RNN and for training the parameters.
The LinearDynamics class is a subclass of ODESimulator utilizing 
intrisic linear dynamics for neuron-level dynamics.

Currently, we don't use CelegansSimulator 
(use torchdiffeq instead of manual Euler) for simulation 
nor do we use KatoDataset, but they are kept for future reference.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
from load import *
from torch import nn

class ODESimulator(nn.Module):
    """Base class for ODE simulators"""
    def __init__(self, connectome_path='data/connectome/White1986', differentiable=True, sheet_name = 0):
        """
        Initialize the base ODE simulator
        
        Parameters:
        - connectome_path: path to the connectivity matrix
        - differentiable: if True, use torchdiffeq for simulation; if False, use scipy's solve_ivp
        """
        super().__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  ('mps' if torch.backends.mps.is_available() else 'cpu'))
        if connectome_path is not None:
            # Use load_synaptic_data to get matrices and neuron_index
            self.differentiable = differentiable
            matrices, neuron_index = load_synaptic_data()
            
            # Create neuron_names from neuron_index to ensure matching order
            self.neuron_names = list(neuron_index.keys())
            
            # Get W from 'EJ' matrix type
            W_np = matrices['EJ']
            self.W = torch.tensor(W_np, dtype=torch.float32, device=self.device)
            self.initial_W = self.W.clone().detach()

            # Get A from 'S' matrix type
            A_np = matrices['S']
            self.A = torch.tensor(A_np, dtype=torch.float32, device=self.device)  
            self.initial_A = self.A.clone().detach()    
            
            self.n_units = len(self.neuron_names)

            
            # Get core neurons from Kato data
            self.sheet_name = sheet_name
            self.core_traces, self.core_traces_dif, neuron_ids, self.core_neuron_names, self.fps = load_kato_data(sheet_name=self.sheet_name)
            #self.fps = round(self.fps, 4)  # Round fps to 2 decimal places

            # Get indices of core neurons in the full network
            self.core_indices = [self.neuron_names.index(n) for n in self.core_neuron_names 
                            if n in self.neuron_names]
            self.m_units = len(self.core_indices)

    def ode_func(self, t, x):
        """
        Abstract method to be implemented by subclasses.
        Should compute dx/dt for the specific ODE system.
        """
        raise NotImplementedError("Subclasses must implement ode_func")
    
    def simulate(self, t_span, x0=None, dt=0.002, method='dopri5'):
        """
        Simulate the ODE system using the selected solver
        
        Parameters:
        - t_span: tuple of (t_start, t_end)
        - x0: initial conditions
        - dt: time step for output points
        - method: integration method ('dopri5', 'rk4', etc.)
        
        Returns:
        - t: time points
        - x: solution trajectories
        """
        if x0 is None:
            raise ValueError("Initial conditions x0 must be provided")
        
        t_eval = torch.arange(t_span[0], t_span[1], dt, device=self.device)
        
        if self.differentiable:
            # Ensure rtol and atol are float32
            rtol = torch.tensor(1e-6, dtype=torch.float32, device=self.device)
            atol = torch.tensor(1e-6, dtype=torch.float32, device=self.device)
            
            # Use torchdiffeq's ODE solver
            x = odeint(
                self.ode_func,
                x0,
                t_eval,
                method=method,
                rtol=rtol,
                atol=atol
            )
        else:
            # Use scipy's solve_ivp for non-differentiable simulation
            def ode_func_scipy(t, y):
                dydt = self.ode_func(t, y)  # y is a numpy array
                return dydt
            
            # Solve the ODE using scipy
            sol = solve_ivp(ode_func_scipy, [t_span[0], t_span[1]], x0.cpu().numpy(), 
                          t_eval=t_eval.cpu().numpy())
            x = torch.tensor(sol.y, dtype=torch.float32).T
        
        return t_eval, x

class LinearDynamics(ODESimulator):
    def __init__(self, connectome_path='data/connectome/White1986', 
                 beta=1.0, tau=5.0, zoomin_factor=None, sheet_name=0,
                 noise_strength=0.1,            
                 d_init_scale=0.5,
                 alpha_cubic_init_mean=0.5,
                 cubic_init_std=0.1,
                 beta_cubic_init_mean= -0.05,
                 gamma_cubic_init_mean = -0.5,
                 synaptic_gain_g=2.5,
                 constrained = True,
                 train_gap_junctions = False):
        """
        Initialize the Cubic Dynamics simulator
        
        Parameters:
        - connectome_path: path to the connectivity matrix
        - beta: Coupling strength
        - tau: Time constant
        - zoomin_factor: Factor for simulation resolution
        - sheet_name: Which sheet to read from Excel files
        - noise_strength: Standard deviation of Gaussian noise (default: 0.0)
        """
        super().__init__(connectome_path, differentiable=True, sheet_name=sheet_name)
        

        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta, device=self.device)))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32, device=self.device))
        self.noise_strength = noise_strength
        
        # Learnable parameters for cubic units
        self.d = nn.Parameter(
            torch.randn(self.n_units, device=self.device, dtype=torch.float32) * d_init_scale
        )
        # Initialize 'alpha_cubic' from a Normal distribution
        self.alpha_cubic = nn.Parameter(
            torch.randn(self.n_units, device=self.device, dtype=torch.float32) * cubic_init_std + alpha_cubic_init_mean
        )
        # Initialize 'beta_cubic' from a Uniform distribution (positive range)
        self.beta_cubic = nn.Parameter(
            torch.rand(self.n_units, device=self.device, dtype=torch.float32) * cubic_init_std + beta_cubic_init_mean
        )

        self.gamma_cubic = nn.Parameter(
            torch.rand(self.n_units, device=self.device, dtype=torch.float32) * cubic_init_std + gamma_cubic_init_mean
        )
        
        # Initialize W as nn.Linear with frozen weights from initial_W
        self.W = nn.Linear(self.n_units, self.n_units, bias=False).to(self.device)
        if not train_gap_junctions:
            self.W.weight = nn.Parameter(self.initial_W.clone(), requires_grad=False)
        else:
            self.W.weight = nn.Parameter(self.initial_W.clone())

        self.A = nn.Linear(self.n_units, self.n_units, bias=False).to(self.device)
        A_mask = (self.initial_A != 0)
        self.A_mask = A_mask.detach().clone().to(dtype=torch.float32, device=self.device)

        # Calculate standard deviation for initialization: g / sqrt(N)
        if self.n_units > 0:
            init_std = synaptic_gain_g / np.sqrt(self.n_units)
        else:
            init_std = 0.01

        # Initialize weights from N(0, std_dev^2)
        initial_A_weights = torch.randn(self.n_units, self.n_units, device=self.device) * init_std # Random Gaussian
        masked_initial_A_weights = initial_A_weights * self.A_mask 
        self.A.weight = nn.Parameter(masked_initial_A_weights)

        total_frames = self.core_traces.shape[1]
        self.total_time = total_frames / self.fps
        self.zoomin_factor = zoomin_factor
        if zoomin_factor is None:
            self.alpha = 1.0 / self.fps
        else:
            self.alpha = 1.0 / (self.fps * zoomin_factor)
    
    @property
    def beta(self):
        return torch.exp(self.log_beta)  # Guaranteed positive
    
    def sigma(self, x):
        """Sigmoid activation function"""
        return torch.tanh(x)
    
    def ode_func(self, t, x):
        """
        Compute dx/dt for the system with gradient clipping to prevent instability
        """
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Compute dynamics for all units
        #f_x = -0.0002 * (x + self.alpha_cubic) * (x + self.beta_cubic) * (x + self.gamma_cubic) + self.d
        f_x = -x + self.d

        # Diffusive coupling (Wij terms)
        x_diff = x.unsqueeze(2) - x.unsqueeze(1)  # Shape: [batch, N, N]
        diffusive = self.beta * torch.sum(self.W.weight * x_diff, dim=2)
        
        # Clip diffusive term to prevent explosion
        diffusive = torch.clamp(diffusive, min=-10.0, max=10.0)

        # Synaptic coupling (Aij terms)
        A_input = self.sigma(x)  # Apply activation before A
        A_term = self.A(A_input)
        
        # Clip synaptic term to prevent explosion
        A_term = torch.clamp(A_term, min=-100.0, max=100.0)

        # Combine all terms and apply time constant
        dxdt = (f_x + diffusive + A_term) / (self.tau + 1e-6)
        
        # Final safety clipping on the entire dxdt
        dxdt = torch.clamp(dxdt, min=-2.0, max=2.0)
        
        return dxdt, f_x / self.tau, diffusive / self.tau, A_term / self.tau
    
    def simulate(self, t_span=None, x0=None, alpha=None, time_steps=None, dt=0.002, method='euler'):
        """
        Simulate the dynamics using Euler integration with explicit time step control and Gaussian noise
        """
        if alpha is None:
            alpha = self.alpha
        if t_span is None:
            t_span = (0, self.total_time)
        if x0 is None:
            # Initialize x0 uniformly between -1 and 1
            x0 = 2 * torch.rand(1, self.n_units, device=self.device) - 1
        
        # Ensure x0 has batch dimension
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
        
        batch_size = x0.shape[0]
        
        # Use explicit time_steps if provided, otherwise calculate from t_span and alpha
        if time_steps is None:
            # Simple calculation without +1
            time_steps = int((t_span[1] - t_span[0]) / alpha)
        
        # Generate evenly spaced time points - exactly time_steps points
        t = torch.linspace(t_span[0], t_span[1], time_steps, device=self.device)
        
        # Recalculate alpha based on actual time points from linspace for accuracy
        actual_alpha = (t[1] - t[0]).item() if time_steps > 1 else alpha # Use calculated step size

        # --- MODIFICATION START: Avoid inplace update ---
        # Initialize list to store states
        states_list = [x0]
        current_state = x0

        # Euler integration loop
        for i in range(time_steps - 1):
            # Calculate derivative using the *current* state
            dx, intrinsic, diffusive, synaptic= self.ode_func(t[i], current_state) # Pass current_state
            """            
            update = actual_alpha * dx
            intrinsic_update = actual_alpha * intrinsic
            diffusive_update = actual_alpha * diffusive
            synaptic_update = actual_alpha * synaptic
            if i % 200 == 0:
                print(f'at time {i}: {update.norm()}, {intrinsic_update.norm()}, {diffusive_update.norm()}, {synaptic_update.norm()}')
            """

            # Add Gaussian noise scaled by noise_strength and sqrt(alpha)
            noise = torch.zeros_like(current_state) # Default to zero noise
            if self.noise_strength > 0:
                noise = torch.randn_like(current_state) * self.noise_strength * torch.sqrt(torch.tensor(actual_alpha, device=self.device))

            # Calculate the *next* state without modifying the list or current_state yet
            next_state = current_state + actual_alpha * dx + noise

            # Check for NaN values and handle them (replace with previous state)
            if torch.isnan(next_state).any():
                print(f"NaN detected at simulation step {i+1}. Replacing with previous state.")
                # Keep current_state as next_state effectively
                next_state = current_state

            # Append the newly computed state to the list
            states_list.append(next_state)
            # Update current_state for the next iteration
            current_state = next_state

        # Stack the list of states into a single tensor after the loop
        # Result shape: [time_steps, batch_size, n_units] (batch_size is likely 1 here)
        states = torch.stack(states_list, dim=0)
        # --- MODIFICATION END ---

        return t, states

    def plot_loss_history(self, train_losses, test_losses, save_path=None):
        """
        Plot the training and test loss history
        
        Parameters:
        - train_losses: list of training loss values
        - test_losses: list of test loss values
        - save_path: path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'b-', alpha=0.8, label='Training Loss')
        plt.plot(test_losses, 'r-', alpha=0.8, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Test Loss History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        # plt.show()


        """
        Load a training checkpoint
        
        Parameters:
        - path: Path to the checkpoint
        - optimizer: Optional optimizer to load state into
        
        Returns:
        - epoch: Last saved epoch number
        - train_losses: Saved training losses
        - test_losses: Saved test losses
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint['train_losses'], checkpoint['test_losses']

    def fit(self, n_epochs=100, learning_rate=0.01, batch_size=1, sequence_length=100, 
            checkpoint_path=None, checkpoint_frequency=10, resume_from=None):
        """
        Train the model with checkpoint support
        
        Parameters:
        - n_epochs: Total number of epochs to train
        - learning_rate: Learning rate for optimizer
        - batch_size: Batch size for training
        - sequence_length: Length of sequences to train on
        - checkpoint_path: Base path for saving checkpoints
        - checkpoint_frequency: How often to save checkpoints (in epochs)
        - resume_from: Path to checkpoint to resume from
        """
        torch.autograd.set_detect_anomaly(True)
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize or load checkpoint
        start_epoch = 0
        train_losses = []
        test_losses = []
        if resume_from is not None:
            start_epoch, train_losses, test_losses = self.load_checkpoint(resume_from, optimizer)
            print(f"Resuming training from epoch {start_epoch}")
        
        # Create datasets
        train_dataset = KatoDataset(
            sequence_length=sequence_length, 
            n_units=self.n_units,
            alpha=self.alpha,
            sheet_name=self.sheet_name  # Training dataset
        )
        test_dataset = KatoDataset(
            sequence_length=sequence_length, 
            n_units=self.n_units,
            alpha= self.alpha,
            zoomin_factor=self.zoomin_factor,
            sheet_name=3  # Test dataset
        )
        test_fps = test_dataset.fps
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print("Starting training...")
        for epoch in range(start_epoch, n_epochs):
            # Training phase
            self.train()
            epoch_train_losses = []
            
            for batch_idx, (target_sequences, weights) in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                target_sequences = target_sequences.to(self.device).permute(2, 0, 1)
                weights = weights.to(self.device).permute(2, 0, 1)
                
                # Initialize x0 from the first timepoint of target sequence
                x0 = target_sequences[0]
                
                t_span = (0, sequence_length / self.fps)
                t, states = self.simulate(t_span=t_span, x0=x0)
                
                if torch.isnan(states).any():
                    print(f"\nNaN detected in training at epoch {epoch}, batch {batch_idx}")
                    raise ValueError("NaN in simulation")
                
                train_loss = torch.mean(weights * (states - target_sequences) ** 2)
                train_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    self.A.weight *= self.A_mask
                
                epoch_train_losses.append(train_loss.item())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                          f"Train Loss: {train_loss.item():.6f}")
            
            # Test phase
            self.eval()
            epoch_test_losses = []
            
            with torch.no_grad():
                for test_sequences, test_weights in test_dataloader:
                    test_sequences = test_sequences.to(self.device).permute(2, 0, 1)
                    test_weights = test_weights.to(self.device).permute(2, 0, 1)
                    
                    x0 = test_sequences[0]
                    t_span = (0, sequence_length / test_fps)
                    t, test_states = self.simulate(t_span=t_span, x0=x0, alpha=1.0 / (self.zoomin_factor * test_fps))
                    
                    test_loss = torch.mean(test_weights * (test_states - test_sequences) ** 2)
                    epoch_test_losses.append(test_loss.item())
            
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Average Train Loss: {avg_train_loss:.6f}")
            print(f"Average Test Loss: {avg_test_loss:.6f}")
            
            # Save checkpoint if requested
            if checkpoint_path and (epoch + 1) % checkpoint_frequency == 0:
                checkpoint_file = f"{checkpoint_path}_current.pt"
                self.save_checkpoint(
                    checkpoint_file, 
                    epoch + 1,
                    train_losses,
                    test_losses,
                    optimizer
                )
        
        # Save final checkpoint
        if checkpoint_path:
            final_checkpoint = f"{checkpoint_path}_final.pt"
            self.save_checkpoint(
                final_checkpoint,
                n_epochs,
                train_losses,
                test_losses,
                optimizer
            )
        
        return train_losses, test_losses
    
    def save_checkpoint(self, path, epoch, train_losses, test_losses, optimizer):
        """
        Save a training checkpoint including new cubic parameters.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'beta': self.beta,
            'tau': self.tau,
            'd': self.d,
            'alpha_cubic': self.alpha_cubic, # ADDED
            'beta_cubic': self.beta_cubic,   # ADDED
            'W': self.W.weight, # Save fixed W if needed elsewhere
            'A': self.A.weight, # Save learnable A weights
            'A_mask': self.A_mask, # Save mask if needed
        }, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """
        Load a training checkpoint including new cubic parameters.
        """
        # Ensure loading onto the correct device
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
             # Load optimizer state, handle potential errors
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state dict: {e}")

        # Load parameters explicitly if needed (though covered by load_state_dict)
        # self.beta = checkpoint['beta']
        # self.tau = checkpoint['tau']
        # self.d = checkpoint['d']
        # self.alpha_cubic = checkpoint['alpha_cubic'] # Ensure these keys exist
        # self.beta_cubic = checkpoint['beta_cubic']   # in the checkpoint file

        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'], checkpoint.get('train_losses', []), checkpoint.get('test_losses', [])

    def save_model(self, path):
        """
        Save the model state dictionary to a file including new cubic parameters.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'beta': self.beta,
            'tau': self.tau,
            'd': self.d,
            'alpha_cubic': self.alpha_cubic, # ADDED
            'beta_cubic': self.beta_cubic,   # ADDED
            'W': self.W.weight,
            'A': self.A.weight,
            'A_mask': self.A_mask,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load the model state dictionary from a file including new cubic parameters.
        """
        # Ensure loading onto the correct device
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
         # Optionally load parameters explicitly if needed for older checkpoints
        # if 'alpha_cubic' in checkpoint: self.alpha_cubic = checkpoint['alpha_cubic']
        # if 'beta_cubic' in checkpoint: self.beta_cubic = checkpoint['beta_cubic']
        print(f"Model loaded from {path}")

class SpikeSimulator(ODESimulator):
    def __init__(self, fish_id='201106',
                 beta=1.0, tau=1.0,
                 noise_strength=0.01,
                 weight_scale=2.0,
                 n_neurons = None):
        """
        Initialize the Spike Simulator
        
        Parameters:
        - fish_id: ID of the fish data to load (default: '201106')
        - beta: Coupling strength
        - tau: Time constant
        - noise_strength: Standard deviation of Gaussian noise
        - weight_scale: Scale factor for random weight initialization
        """
        # Load fish spike data to get number of neurons
        spike_data = load_fish_spike_data(ifish=fish_id, n_neurons=n_neurons)
        n_units = spike_data.shape[0]  # Number of neurons from spike data
        
        # Initialize base class without connectome
        super().__init__(connectome_path=None)
        
        # Override n_units from base class
        self.n_units = n_units
        
        self.log_beta = nn.Parameter(torch.log(torch.tensor(beta, device=self.device)))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32, device=self.device))
        self.noise_strength = noise_strength
        
        # Initialize synaptic weights randomly
        # Using Xavier/Glorot initialization scaled by weight_scale
        self.W = nn.Linear(self.n_units, self.n_units, bias=False).to(self.device)
        nn.init.xavier_normal_(self.W.weight, gain=weight_scale)
        
        # Store spike data for reference
        self.spike_data = spike_data
    
    @property
    def beta(self):
        return torch.exp(self.log_beta)
    
    def sigma(self, x):
        """Sigmoid activation function"""
        return torch.relu(x)
    
    def ode_func(self, t, x):
        """
        Compute dx/dt for the spike-based system
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Compute synaptic input
        synaptic_input = self.W(x)
        
        # Add noise if enabled
        if self.noise_strength > 0:
            noise = torch.randn_like(x) * self.noise_strength
            synaptic_input = synaptic_input + noise
        
        # Compute membrane potential dynamics
        dxdt = (-x + self.beta * self.sigma(synaptic_input)) / self.tau

        dxdt = torch.clamp(dxdt, min=-1.0, max=1.0)
        
        return dxdt
    
    def simulate(self, t_span=None, x0=None, time_steps=None):
        """
        Simulate the dynamics using Euler integration
        """
        if x0 is None:
            # Initialize x0 uniformly between -1 and 1
            x0 = 2 * torch.rand(1, self.n_units, device=self.device) - 1
        
        # Ensure x0 has batch dimension
        if len(x0.shape) == 1:
            x0 = x0.unsqueeze(0)
        
        # Use explicit time_steps if provided, otherwise calculate from t_span and alpha
        if time_steps is None:
            return None
        
        # Generate evenly spaced time points - exactly time_steps points
        t = torch.linspace(t_span[0], t_span[1], time_steps, device=self.device)
        step_size = (t[1] - t[0]).item() 

        states_list = [x0]
        current_state = x0

        # Euler integration loop
        for i in range(time_steps - 1):
            # Calculate derivative using the *current* state
            dxdt = self.ode_func(t[i], current_state) 

            # Calculate the *next* state without modifying the list or current_state yet
            next_state = current_state + step_size * dxdt 

            # Check for NaN values and handle them (replace with previous state)
            if torch.isnan(next_state).any():
                print(f"NaN detected at simulation step {i+1}. Replacing with previous state.")
                # Keep current_state as next_state effectively
                next_state = current_state

            # Append the newly computed state to the list
            states_list.append(next_state)
            # Update current_state for the next iteration
            current_state = next_state

        states = torch.stack(states_list, dim=0)
        return t, states

class KatoDataset(torch.utils.data.Dataset):
    """Dataset class for Kato et al. neural activity data"""
    def __init__(self, sequence_length, n_units, alpha, zoomin_factor = None, normalize=True, sheet_name = 0):
        """
        Initialize the Kato dataset
        
        Parameters:
        - T - sequence_length: length of sequences to return
        - n - n_units: total number of units in the network
        - m - number of latent variables 
        - normalize: if True, normalize data to [0, 1] range
        """
        # Load Kato data
        traces, traces_dif, neuron_ids, self.neuron_names, self.fps = load_kato_data(sheet_name=sheet_name)
        self.fps = round(self.fps, 2)
        
        # Convert to torch tensors and truncate to sequence_length
        core_traces = torch.tensor(traces[:, :sequence_length], dtype=torch.float32)
        
        # Compute mean and std per row
        row_means = core_traces.mean(dim=1, keepdim=True)  # Shape: (num_neurons, 1)
        row_stds = core_traces.std(dim=1, keepdim=True)    # Shape: (num_neurons, 1)

        # Normalize each row
        core_traces = (core_traces - row_means) / row_stds
        if zoomin_factor is not None:
            alpha = 1 / (zoomin_factor * self.fps)

        # Calculate expanded timepoints
        total_time = sequence_length / self.fps
        n_expanded_points = int(total_time / alpha)
        points_per_frame = int(1 / (self.fps * alpha))


        
        print(f"Original points: {sequence_length}")
        print(f"Expanded points: {n_expanded_points}")
        print(f"Points per frame: {points_per_frame}")
        
        # Create expanded traces
        expanded_traces = torch.zeros(len(core_traces), n_expanded_points)
        for i in range(sequence_length):
            idx = i * points_per_frame
            expanded_traces[:, idx] = core_traces[:, i]
        
        # Expand to full network size
        self.traces = torch.zeros(n_units, n_expanded_points, dtype=torch.float32)
        self.traces[:len(core_traces)] = expanded_traces
        
        # Create weight matrix [n_units, n_timepoints]
        self.weights = torch.zeros(n_units, n_expanded_points)
        # Set weights for core neurons at actual data points
        for i in range(sequence_length):
            idx = i * points_per_frame
            self.weights[:len(core_traces), idx] = 1.0
        
        self.sequence_length = n_expanded_points
        self.n_neurons = len(self.neuron_names)
        self.m_units = len(core_traces)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.traces, self.weights


def main():
    # Initialize model
    simulator = SpikeSimulator()
    time_span = (0.0, 0.1)
    steps = 3
    t, states = simulator.simulate(t_span=time_span, time_steps=steps)
   
if __name__ == "__main__":
    main() 