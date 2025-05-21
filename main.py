"""
Main entry point for the C. elegans neural dynamics training pipeline.

This script orchestrates the training process, leveraging the TrialPCA model
to fit neural dynamics to PCA-reduced C. elegans activity data. 

It allows for comparing models trained with and without connectome-based constraints.
Also, it contains the post-training visualization methods.

Running main() always compares constrained versus unconstrained
but everything else is tune-able.
"""

from pathlib import Path
from trial_pca import *
from vis_average_loss import load_and_average_metrics
from vis_average_pc import average_pc_trajectories

def main():
    """
    Main function to run the training and visualization pipeline
    with hardcoded parameters.
    """

    # IMPORTANT: set these manually!
    which_worm = 0 # which data sheet to use?
    num_runs = 5 # how many times to repeat training with randomness?
    num_epochs = 100
    training_volume = 300 # number of frames in the training set 
    test_volume = 150 # number of frames in the training set 
    train_gap_junctions = True

    # you don't have to modify this!
    zoomin_factor_for_numerical_stability = 5
    
    # many more hyper-params to tune
    config = {
        'sheet_name': which_worm,
        'num_runs': num_runs, # Number of independent training runs
        'base_dir_template': "checkpoints/run{run_j}",
        'base_figs_template': "figs/run{run_j}",

        # --- Data Parameters ---
        'data_dir': 'data/activity/WT_NoStim',
        'connectome_path': 'data/connectome/White1986',
        'latent_dim': 8,
        'sequence_length': training_volume,
        'test_seq_length': test_volume,
        'detrend': True,
        'zoomin_factor': zoomin_factor_for_numerical_stability,

        # --- Training Hyperparameters ---
        'n_epochs': num_epochs, 
        'learning_rate': 2e-2,
        'weight_decay': 1e-4,
        'lambda_cov': 3e-2,
        'lambda_corr': 1e-4,
        'scheduler_patience': 20,
        'scheduler_factor': 0.9,
        'checkpoint_frequency': 10,

        # --- Model/Simulator Initial Parameters ---
        'beta_init': 0.02,
        'tau_init': 3.0,
        'noise_init': 0.0,
        'synaptic_gain_init': 7.5,
        'train_gap_junctions': train_gap_junctions,

        # --- Post-Training Visualization Control ---
        'run_avg_loss_vis': True, # Set to True to run this visualization
        'run_avg_pc_vis': True,   # Set to True to run this visualization

        # Parameters for average loss visualization
        'avg_loss_num_runs_vis': num_runs, # Number of runs to average (should match num_runs or be less)
        'avg_loss_epochs_vis': num_epochs, # Number of epochs to display

        # Parameters for average PC trajectory visualization
        'avg_pc_num_runs_vis': num_runs, # Number of runs to average
        'avg_pc_seq_len_vis': training_volume, 
        'avg_pc_zoomin_vis': zoomin_factor_for_numerical_stability, 
        'avg_pc_save_path_vis': "averaged_trajectories.html"
    }

    print("Starting training pipeline with the following parameters:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("-" * 30)

    # --- Main Training Loop ---
    for i in range(config['num_runs']):
        run_num = i # Actual run number for directory naming
        print(f"\n===== Starting Run {run_num + 1}/{config['num_runs']} for Trial {config['sheet_name']} =====")

        current_checkpoint_path_str = config['base_dir_template'].format(run_j=run_num)
        current_fig_path_str = config['base_figs_template'].format(run_j=run_num)

        # Create output directories for the current run
        Path(current_checkpoint_path_str).parent.mkdir(parents=True, exist_ok=True)
        Path(current_fig_path_str).parent.mkdir(parents=True, exist_ok=True)

        # Path for the specific trial and comparison within the run
        trial_checkpoint_base = Path(current_checkpoint_path_str) / f"trial_{config['sheet_name']}_compare"
        trial_fig_base = Path(current_fig_path_str) / f"trial_{config['sheet_name']}_compare"

        compare_constrain(
            sheet_name=config['sheet_name'],
            latent_dim=config['latent_dim'],
            sequence_length=config['sequence_length'],
            test_seq_length=config['test_seq_length'],
            detrend=config['detrend'],
            zoomin_factor=config['zoomin_factor'],
            n_epochs=config['n_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            lambda_cov=config['lambda_cov'],
            lambda_corr=config['lambda_corr'],
            scheduler_patience=config['scheduler_patience'],
            scheduler_factor=config['scheduler_factor'],
            checkpoint_frequency=config['checkpoint_frequency'],
            beta_init=config['beta_init'],
            tau_init=config['tau_init'],
            noise_init=config['noise_init'],
            synaptic_gain_init=config['synaptic_gain_init'],
            data_dir=config['data_dir'],
            connectome_path=config['connectome_path'],
            base_checkpoint_path=str(trial_checkpoint_base),
            base_fig_path=str(trial_fig_base),
            train_gap_junctions=config['train_gap_junctions']
        )
        print(f"===== Run {run_num + 1}/{config['num_runs']} Finished =====")

    print("\nAll training runs completed.")

    # --- Post-Training Visualizations ---
    if config['run_avg_loss_vis']:
        print("\n--- Generating Average Loss Visualization ---")
        avg_loss_base_dir = Path(config['base_dir_template'].format(run_j=0)).parent
        load_and_average_metrics(
            trial_x=config['sheet_name'],
            base_dir=str(avg_loss_base_dir),
            num_runs=config['avg_loss_num_runs_vis'],
            epochs=config['avg_loss_epochs_vis']
        )
        print("Average loss visualization complete.")

    if config['run_avg_pc_vis']:
        print("\n--- Generating Average PC Trajectory Visualization ---")
        avg_pc_base_dir = Path(config['base_dir_template'].format(run_j=0)).parent
        average_pc_trajectories(
            trial_x=config['sheet_name'],
            base_dir=str(avg_pc_base_dir),
            num_runs=config['avg_pc_num_runs_vis'],
            sequence_length=config['avg_pc_seq_len_vis'],
            zoomin_factor=config['avg_pc_zoomin_vis'],
            save_path=config['avg_pc_save_path_vis']
        )
        print(f"Average PC trajectory visualization saved to {config['avg_pc_save_path_vis']}")

    print("\nPipeline finished.")

if __name__ == "__main__":
    main()