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
from model import *
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

# --- New function to run the SpikeCovarianceModel ---
def run_spike_covariance_experiment():
    """
    Runs a training experiment for the SpikeCovarianceModel.
    """
    print("\n===== Starting Spike Covariance Model Experiment =====")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Parameters for the SpikeCovarianceModel experiment
    fish_id = '201106'  # Example fish ID, ensure data exists for this ID
    fish_figs_dir = 'fish-figs/'
    sequence_length = 500 # Number of timepoints for simulation and target data
    n_neurons = None
    learning_rate = 1e-3 # Adjusted learning rate
    lambda_activity = 1e-5
    num_epochs = 200
    start_time = 0.0
    end_time = sequence_length / 10.0


    tau_init = 1.0
    bias_init_std = 0.5
    noise_strength = 1e-2
    weight_scale = 2.0

    print(f"Experiment Parameters:")
    print(f"  Fish ID: {fish_id}")
    print(f"  Sequence Length: {sequence_length}")
    print(f"  Number of Neurons: {n_neurons}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Bias Init Std: {bias_init_std}, Tau Init: {tau_init}, Lambda: {lambda_activity}, Noise: {noise_strength}")
    print("-" * 30)

    # Instantiate the SpikeCovarianceModel
    try:
        model = SpikeCovarianceModel(
            fish_id=fish_id,
            sequence_length=sequence_length,
            tau_init=tau_init,
            bias_init_std=bias_init_std,
            noise_strength=noise_strength,
            weight_scale=weight_scale,
            learning_rate=learning_rate,
            device=device,
            n_neurons=n_neurons,
            lambda_activity_mse=lambda_activity
        )
        print("SpikeCovarianceModel instantiated successfully.")
        # # Example: Plot activity for the first 2 seconds for the first 3 neurons
        plot_activity_comparison(
            model=model,
            start_time_sec=start_time,
            end_time_sec=end_time,
            max_neurons_to_plot=25, # Or use this to plot the first N
            save_path=fish_figs_dir + "activity_comparison_prior.png"
        )
    except Exception as e:
        print(f"Error instantiating SpikeCovarianceModel: {e}")
        import traceback
        traceback.print_exc()
        return

    # Train the model
    print("Starting training...")
    try:
        training_history = model.fit(
            n_epochs=num_epochs,
            save_path_prefix=fish_figs_dir
        )
        print("Training finished.")
        plot_activity_comparison(
            model=model,
            start_time_sec=start_time,
            end_time_sec=end_time,
            max_neurons_to_plot=25, # Or use this to plot the first N
            save_path=fish_figs_dir + "activity_comparison_posterior.png"
        )

    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()

    print("===== Spike Covariance Model Experiment Finished =====")

if __name__ == "__main__":
    run_spike_covariance_experiment()