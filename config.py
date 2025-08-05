"""
This module provides the hyperparameters necessary for work with the model.
"""

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name', default=None, help='logs and checkpoints will be saved in corresponding subfolders having this name')
args = parser.parse_args()

if args.exp_name == None:
    raise RuntimeError('--exp_name argument must be specified')


# Model Configuration
class HifiGanConfig:
    """
    This class hols all the parameters for training run. See attribute comments for details.
    """
    # Path of downloaded AudioSet directory
    audio_path = "/scratch/vb00479/audioset_balanced_22k/train"
    # Path of selected audio clips that will be used to print spectrograms and generated clips in TensorBoard every `visualization_interval` steps
    audio_to_visualize_path = "/scratch/vb00479/audioset_balanced_22k/visuals"
    
    # Target sampling rate. Each clip will be resampled at this sampling rate
    sampling_rate = 16000
    
    # The length of audio segments to use for training and validation (in samples)
    segment_size = 8192

    # Mel-Spectrogram parameters for the new sampling rate
    n_fft = 1024          # FFT window size
    n_mels = 64           # Number of Mel bands
    hop_size = 256        # Hop size between frames
    win_size = 1024       # Window size
    fmin = 0              # Minimum frequency
    fmax = 8000           # Maximum frequency

    # Training parameters
    steps = 100000
    batch_size = 16
    learning_rate_g = 0.0002
    learning_rate_d = 0.0002
    adam_b1 = 0.8
    adam_b2 = 0.99
    lr_decay = 0.999
    seed = 1234

    # Validation parameters
    validation_split_ratio = 0.05 # Adjusts train/val split
    validation_interval = 1000 # Validate model every specified number of steps (i.e. interval)
    num_batches_to_val = 30 # How many batches of `batch_size` to use for validation

    # Checkpoint and Logging
    run_name = args.exp_name.replace(' ', '_')

    checkpoint_dir = "/scratch/vb00479/audioset_balanced_22k/checkpoints" # Parent directory for checkpoints
    checkpoint_path = os.path.join(checkpoint_dir, run_name) # Subdirectory to save model checkpoints for a specific run
    checkpoint_interval = 20000 # How often to save checkpoints
    log_interval = 100 # Console logging interval
    visualization_interval = 10000 # How often to save generated audio clips and their spectrograms from `audio_to_visualize_path` directory
    log_dir = "./hifi-gan_runs" # Parent directory for TensorBoard logs
    tensorboard_path = os.path.join(log_dir, run_name) # Subdirectory to save TensorBoard logs for a specific run


# Save hyperparameter values for TensorBoard logging
hparam_dict = {k: v for k, v in vars(HifiGanConfig).items() if not k.startswith('__')}