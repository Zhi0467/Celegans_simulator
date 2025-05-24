import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def load_and_average_metrics(trial_x, base_dir='checkpoints', num_runs=10, epochs = 100):
    # Initialize storage for metrics
    constrained_metrics = {
        'latent_mse': [],
        'neural_mse': [],
        'train_cov_sim': [],
        'test_cov_sim': [],
    }
    unconstrained_metrics = {
        'latent_mse': [],
        'neural_mse': [],
        'train_cov_sim': [],
        'test_cov_sim':[]
    }
    
    for run_j in range(num_runs):
        # Construct paths for constrained and unconstrained checkpoints
        constrained_path = os.path.join(base_dir, f'run{run_j}', f'trial_{trial_x}_compare_constrained:True_final.pt')
        unconstrained_path = os.path.join(base_dir, f'run{run_j}', f'trial_{trial_x}_compare_constrained:False_final.pt')
        
        # Load constrained model metrics
        if os.path.exists(constrained_path):
            try:
                ckpt = torch.load(constrained_path, map_location='cpu')
                constrained_metrics['latent_mse'].append(ckpt['train_losses_mse'][:epochs])
                constrained_metrics['neural_mse'].append(ckpt['train_losses_neural_mse'][:epochs])
                constrained_metrics['train_cov_sim'].append(ckpt['train_cov_similarities'][:epochs])
                constrained_metrics['test_cov_sim'].append(ckpt['test_cov_similarities'][:epochs])
            except Exception as e:
                print(f"Error loading constrained checkpoint for run {run_j}: {e}")
        else:
            print(f"Constrained checkpoint not found for run {run_j}")
        
        # Load unconstrained model metrics
        if os.path.exists(unconstrained_path):
            try:
                ckpt = torch.load(unconstrained_path, map_location='cpu')
                unconstrained_metrics['latent_mse'].append(ckpt['train_losses_mse'][:epochs])
                unconstrained_metrics['neural_mse'].append(ckpt['train_losses_neural_mse'][:epochs])
                unconstrained_metrics['train_cov_sim'].append(ckpt['train_cov_similarities'][:epochs])
                unconstrained_metrics['test_cov_sim'].append(ckpt['test_cov_similarities'][:epochs])
            except Exception as e:
                print(f"Error loading unconstrained checkpoint for run {run_j}: {e}")
        else:
            print(f"Unconstrained checkpoint not found for run {run_j}")
    
    # Function to compute average and std
    def compute_stats(metrics_list):
        if not metrics_list:
            return None, None
        max_length = max(len(m) for m in metrics_list)
        padded = [m + [np.nan] * (max_length - len(m)) for m in metrics_list]
        arr = np.array(padded)
        avg = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        return avg, std
    
    # Compute average and std for each metric
    constrained_avg, constrained_std = {}, {}
    unconstrained_avg, unconstrained_std = {}, {}
    for key in ['latent_mse', 'neural_mse', 'train_cov_sim', 'test_cov_sim']:
        c_avg, c_std = compute_stats(constrained_metrics[key])
        u_avg, u_std = compute_stats(unconstrained_metrics[key])
        constrained_avg[key] = c_avg
        constrained_std[key] = c_std
        unconstrained_avg[key] = u_avg
        unconstrained_std[key] = u_std
    
    # Plotting
    plt.figure(figsize=(18, 5))
    metrics = ['latent_mse', 'neural_mse', 'train_cov_sim', 'test_cov_sim']
    titles = ['Train Latent MSE', 'Train Neural MSE', 'Train Neural Covariance Similarity', 'Test Neural Covariance Similarity']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 4, i+1)
        steps = np.arange(1, epochs + 1)
        # Plot constrained
        if constrained_avg[metric] is not None:
            plt.plot(steps, constrained_avg[metric], label='Constrained', color='blue')
            plt.fill_between(steps, 
                             constrained_avg[metric] - constrained_std[metric], 
                             constrained_avg[metric] + constrained_std[metric], 
                             color='blue', alpha=0.2)
        
        # Plot unconstrained
        if unconstrained_avg[metric] is not None:
            plt.plot(steps, unconstrained_avg[metric], label='Unconstrained', color='red')
            plt.fill_between(steps, 
                             unconstrained_avg[metric] - unconstrained_std[metric], 
                             unconstrained_avg[metric] + unconstrained_std[metric], 
                             color='red', alpha=0.2)
        
        plt.xlabel('Epoch')
        plt.title(title)
        if metric in ['latent_mse', 'neural_mse']:
            plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    plt.suptitle(f"Training Statistics on Trial {trial_x} Average over {num_runs} runs")
    plt.tight_layout()
    plt.savefig(f'avg_loss_trial_{trial_x}_with_num_runs_{num_runs}.png')
