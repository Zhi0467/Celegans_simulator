# Neural Simulators 

## C. elegans Neural Dynamics Training Pipeline

This project provides a pipeline to train and evaluate models of C. elegans neural dynamics. The `main.py` script is the primary entry point for orchestrating the training process, comparing models with and without connectome-based constraints, and generating post-training visualizations.

Data folder is not provided in this repo as it'll be too chunky, you can download from original White and Zimmer paper or ask me for it.

### Prerequisites

1.  **Python Environment**: Ensure you have a Python environment (e.g., Python 3.8+) set up.
2.  **Dependencies**: Install the necessary Python packages. You can typically do this using a `requirements.txt` file (if provided) with pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key libraries include `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `plotly`, and `tqdm`.
3.  **Data (not included in the repo to keep it slim)**:
    * Neural activity data (e.g., `IDs.xlsx`, `traces.xlsx`, `tracesDif.xlsx`, `fps.xlsx`) should be placed in a directory specified by `data_dir` in the `config` (default: `data/activity/WT_NoStim/`).
    * Connectome data (e.g., `connectome_EJ.csv`, `Celeganconnect.xls`) should be in a directory specified by `connectome_path` (default: `data/connectome/White1986/`).

### Configuration

Parameters for the training pipeline are primarily configured within the `main()` function in `main.py`.

Key parameters to adjust at the beginning of the `main()` function:
* `num_runs`: Number of independent training runs.
* `num_epochs`: Number of training epochs per run.
* `training_volume`: Number of frames in the training set.
* `test_volume`: Number of frames in the test set.
* `train_gap_junctions`: Boolean to determine if gap junction weights in the model should be trainable.

Further detailed hyperparameters are located in the `config` dictionary within `main.py`. Users can modify these values directly in the script before running. Important `config` settings include:

* `sheet_name`: Specifies which sheet from the neural activity Excel files to use.
* `data_dir`, `connectome_path`: Paths to the input data.
* `latent_dim`: The dimensionality of the latent space for PCA.
* `sequence_length`, `test_seq_length`: Length of training and testing sequences.
* `n_epochs`, `learning_rate`, `weight_decay`, `lambda_cov`, `lambda_corr`: Hyperparameters for the training process.
* `beta_init`, `tau_init`, `noise_init`, `synaptic_gain_init`: Initial parameters for the neural dynamics model.
* `run_avg_loss_vis`, `run_avg_pc_vis`: Booleans to enable/disable post-training average loss and PC trajectory visualizations.

### Running the Script

To run the training pipeline, execute `main.py` from your terminal:

```bash
python minimal_code/main.py