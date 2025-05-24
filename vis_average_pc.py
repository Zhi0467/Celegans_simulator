import os
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from model import *

def average_pc_trajectories(trial_x, base_dir='checkpoints', num_runs=10, 
                           sequence_length=300, zoomin_factor=5, save_path=None):

    common_params = {
        'beta': 0.02, 'tau': 3.0, 'zoomin_factor': zoomin_factor,
        'sheet_name': trial_x, 'data_dir': 'data/activity/WT_NoStim',
        'connectome_path': 'data/connectome/White1986', 'detrend': True,
        'synaptic_gain': 7.5, 'noise_strength': 0.0 # For deterministic simulation
    }

    # 2. Load dataset for actual PCA trajectory
    dataset = IdentifiedPCA(
        latent_dim=8, zoomin_factor=zoomin_factor,
        sequence_length=sequence_length, test_seq_length=0,
        sheet_name=trial_x, data_dir=common_params['data_dir'],
        connectome_path=common_params['connectome_path'],
        detrend=common_params['detrend']
    )

    # 3. Storage for trajectories
    all_constrained = []
    all_unconstrained = []
    
    # 4. Load and process each run
    for run_j in tqdm(range(num_runs), desc="Processing runs"):
        # Load constrained model
        constrained_path = os.path.join(
            base_dir, f'run{run_j}', 
            f'trial_{trial_x}_compare_constrained:True_final.pt'
        )
        if not os.path.exists(constrained_path):
            continue
            
        try:
            # Load checkpoint and simulate
            ckpt = torch.load(constrained_path, map_location='cpu')
            model = create_model_from_checkpoint(ckpt, common_params, constrained=True)
            constrained_traj = simulate_trajectory(model, dataset, sequence_length, zoomin_factor)
            all_constrained.append(constrained_traj)
        except Exception as e:
            print(f"Error with constrained run {run_j}: {e}")

        # Repeat for unconstrained
        unconstrained_path = os.path.join(
            base_dir, f'run{run_j}', 
            f'trial_{trial_x}_compare_constrained:False_final.pt'
        )
        if not os.path.exists(unconstrained_path):
            continue
            
        try:
            ckpt = torch.load(unconstrained_path, map_location='cpu')
            model = create_model_from_checkpoint(ckpt, common_params, constrained=False)
            unconstrained_traj = simulate_trajectory(model, dataset, sequence_length, zoomin_factor)
            all_unconstrained.append(unconstrained_traj)
        except Exception as e:
            print(f"Error with unconstrained run {run_j}: {e}")

    # 5. Average trajectories
    avg_constrained = np.nanmean(np.array(all_constrained), axis=0) if all_constrained else None
    avg_unconstrained = np.nanmean(np.array(all_unconstrained), axis=0) if all_unconstrained else None

    # 6. Generate plot
    fig = create_3d_plot(dataset, avg_constrained, avg_unconstrained, sequence_length)
    
    if save_path:
        pio.write_html(fig, save_path)
        print(f"Averaged trajectories saved to {save_path}")
    return fig

def create_model_from_checkpoint(ckpt, common_params, constrained):
    """Instantiate model from checkpoint data"""
    # FIX 1: Get current device first
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = TrialPCA(
        latent_dim=ckpt.get('latent_dim', 8),
        beta=common_params['beta'],
        tau=common_params['tau'],
        zoomin_factor=common_params['zoomin_factor'],
        sheet_name=common_params['sheet_name'],
        data_dir=common_params['data_dir'],
        connectome_path=common_params['connectome_path'],
        sequence_length=ckpt.get('sequence_length', 300),
        test_seq_length=0,
        detrend=common_params['detrend'],
        synaptic_gain=common_params['synaptic_gain'],
        constrained=constrained
    )
    model.dynamics_model.load_state_dict(ckpt['dynamics_model_state'])
    model.observed_projection.load_state_dict(ckpt['observed_projection_state'])
    
    # FIX 2: Move entire model to target device
    model.to(device)
    return model

def simulate_trajectory(model, dataset, sequence_length, zoomin_factor):
    """Run simulation and return latent states"""
    model.eval()
    # FIX 3: Get model's device dynamically
    device = next(model.parameters()).device
    
    # FIX 4: Create tensor on model's device
    x0_sim = torch.zeros(1, model.n_sim_neurons, device=device)
    
    # FIX 5: Move dataset values to model's device
    for dataset_idx, sim_idx in model.dataset_to_sim_idx.items():
        if dataset_idx < dataset.x0.shape[1]:
            x0_sim[0, sim_idx] = dataset.x0[0, dataset_idx].to(device)

    with torch.no_grad():
        _, _, _, latent_states = model.forward(
            x0=x0_sim,
            t_span=(0, sequence_length/dataset.fps),
            time_steps=sequence_length*zoomin_factor
        )
    return latent_states.squeeze(1).cpu().numpy()

def create_3d_plot(dataset, avg_constrained, avg_unconstrained, sequence_length):
    """Create 3D plot with averaged trajectories"""
    # Get actual PCA trajectory
    actual_latent = dataset.latent_traces[:, :sequence_length].cpu().numpy()
    time_actual = np.linspace(0, sequence_length/dataset.fps, actual_latent.shape[1])

    fig = go.Figure()

    # Plot actual data
    fig.add_trace(go.Scatter3d(
        x=actual_latent[0], y=actual_latent[1], z=actual_latent[2],
        mode='lines', line=dict(color='red', width=4),
        name='Actual Data (PCA)'
    ))

    # Plot averaged constrained trajectory
    if avg_constrained is not None and avg_constrained.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=avg_constrained[:, 0], y=avg_constrained[:, 1], z=avg_constrained[:, 2],
            mode='lines', line=dict(color='blue', width=4),
            name='Averaged Constrained'
        ))

    # Plot averaged unconstrained trajectory
    if avg_unconstrained is not None and avg_unconstrained.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=avg_unconstrained[:, 0], y=avg_unconstrained[:, 1], z=avg_unconstrained[:, 2],
            mode='lines', line=dict(color='green', width=4),
            name='Averaged Unconstrained'
        ))

    fig.update_layout(
        title=f'Averaged Latent Trajectories (Trial 1)',
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3',
            aspectmode='cube'
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        legend_title_text='Trace Type'
    )
    fig.write_image('trial1.png')
    return fig
