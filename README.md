# Conditional HiFi-GAN for Dynamic Sampling Rate Control

A PyTorch implementation of a conditional [HiFi-GAN vocoder](https://arxiv.org/abs/2010.05646) capable of generating high-fidelity audio at multiple, user-defined sampling rates.

This is a repository for the code of master's dissertation "Enhancing Controllability of Text-to-Audio Generation Models" completed as a part of 'MSc in Artificial Intelligence' at the University of Surrey.

## Table of Contents

* [Introduction](#introduction)

* [Key Features](#key-features)

* [Project Structure](#project-structure)

* [Getting Started](#getting-started)

    * [Prerequisites](#prerequisites)

    * [Installation](#installation)

* [Usage](#usage)

    * [0. Uploading AudioSet](#0-uploading-audioset)

    * [1. Dataset Preparation](#1-dataset-preparation)

    * [2. Configuration](#2-configuration)

    * [3. Training](#3-training)
    
    * [4. Evaluation](#4-evaluation)

* [Results](#results)

* [License](#license)

* [Acknowledgments](#acknowledgments)

## Introduction

Modern text-to-audio models can generate highly realistic audio but often lack fine-grained control over essential audio properties. A critical limitation is their inability to produce audio at variable sampling rates, restricting their use in applications with diverse requirements (e.g., 8kHz for telephony vs. 48kHz for music).

This project addresses that gap by introducing a conditional vocoder based on the state-of-the-art HiFi-GAN architecture. The model is conditioned on a target sample rate, allowing it to dynamically control the bandwidth and fidelity of the output waveform.

## Key Features

* **Dynamic Sampling Rate Control**: Generate audio at multiple discrete sampling rates (e.g., 11.025kHz, 16kHz, 22.05kHz, 44.1kHz, 48kHz) from a single trained model

* **FiLM Conditioning**: Utilizes [Feature-wise Linear Modulation (FiLM)](https://arxiv.org/abs/1709.07871) layers to inject the sample rate conditioning signal directly into the generator's residual blocks and transform feature maps of a synthesized waveform

* **Disentangled Training Strategy**: Employs a novel data loading approach that separates the audio's content (high-resolution spectrogram) from its rendering style (target sample rate) to ensure physically accurate synthesis

* **Novel Spectral Cutoff Loss**: Includes a specialized loss function that directly penalizes the generation of incorrect high-frequency content, forcing the model to adhere to the Nyquist limit of the target rate

* **High-Fidelity Output**: Maintains audio quality comparable to the baseline non-conditional HiFi-GAN

The following image depicts the suggested architecture of FiLM-conditioned Residual Block:

![Architecture](/pics/architecture.png)

## Project Structure

```
.
├── datasets/                       # Data loading and pre-processing
│   └── audiodataset.py
├── losses/                         # Custom loss functions
│   ├── adversarial_loss.py
│   └── spectral_cutoff_loss.py
├── models/                         # Core ANN model architectures 
│   ├── film_resblock.py
│   ├── generator.py
│   ├── mpd.py
│   ├── msd.py
│   └── sample_rate_embedding.py
├── scripts/                        # Helper scripts for evaluation and data handling
│   ├── download_dataset.py
│   ├── objective_eval.py
│   └── subjective_eval.py
├── trainers/                       # The main training loop logic
│   └── trainer.py
├── utils/                          # Utility functions for spectrogram handling, etc.
│   ├── init_weights.py
│   └── specs.py
├── config.py                       # Configuration file 
├── main.py                         # Script for running training 
├── Makefile                        # Command shortcuts for common tasks
└── requirements.txt                # Project dependencies
```

## Getting Started

Follow these steps to set up the project environment.

### Prerequisites

* Python 3.10+
* PyTorch 2.0+
* CUDA-enabled GPU (recommended for training)

### Installation

**1. Clone the repository**:
```bash
git clone https://github.com/He11Coder/controllable-audioldm-msc-project.git
cd controllable-audioldm-msc-project
```

**2. Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### 0. Uploading AudioSet

If you would like to use [AudioSet](https://research.google.com/audioset/) as the main dataset, there is a script `scripts/download_dataset.py` in the current reposiroty which can be utilized to upload AudioSet. Follow the comments in this file to specify target directories.

### 1. Dataset Preparation

Assign a path to your training audio files (e.g., `.wav` format) to `audio_path` variable inside `HifiGanConfig` class in `config.py`. The data loader will recursively search for all .wav files in this folder and will use them to feed the model during training.

For visualization and post-training evaluation, place small, fixed sets of audio files in preferred directories and follow the next step. Validation set will ba automatically created from the files in `audio_path`.

### 2. Configuration

All hyperparameters for training and model architecture are located in `config.py`. Before running, review and adjust the following key parameters:

* `audio_path`: path to your main training dataset

* `audio_to_visualize_path`: path to your visualization dataset (visualization is done periodically during training and the results are logged via TensorBord)

* `native_sampling_rate`: the sampling rate of the audio files in your dataset

* `supported_sampling_rates`: the list of sample rates the model will be trained on

* `steps`: total number of training steps (number of batches processed during training)

* `batch_size`: the batch size for training

* `checkpoint_dir`: parent directory for the subdirectories where model checkpoints will be saved for each training run

* `log_dir`: parent directory for the subdirectories where TensorBoard log files will be saved for each training run

For the rest of the parameters (advanced ones) see `config.py` and follow the comments.

### 3. Training

To start a training run, you may execute `main.py`:
```bash
python3 main.py --exp_name='run_name'
```

or you can use one of the commands from `Makefile` to do so:
```bash
make run-train EXP='run_name'
```

Training progress, including loss curves, audio samples, and spectrograms, can be monitored in real-time using TensorBoard:
```bash
tensorboard --logdir='hifi-gan_runs'
```

or use the corresponding make-command:
```bash
make run-tboard
```

### 4. Evaluation

<ins>Objective Evaluation</ins>

To conduct an objective evaluation of a trained model, use `scripts/objective_eval.py` script. This script will generate audio for a given set of files and calculate objective metrics (FAD, IS and KL Divergence).

Open `scripts/objective_eval.py` and find the required command line arguments at the bottom of the script. Make sure you have included all of them in the following script and run evaluation: 
```bash
cd scripts/
python3 objective_eval.py
```

The results will be saved to a `.json` file in appropriate directory.

<ins>Subjective Evaluation</ins>

You can do a subjective evaluation, i.e., generate some audio clips at different (potentially unseen) sampling rates and save their spectrograms to visually validate the results.

Open `scripts/subjective_eval.py` if you need more details or comments. To start evaluation, run:
```bash
cd scripts/
python3 subjective_eval.py --exp_name='run_name'
```

`--exp_name` argument must match the name of the experiment (i.e., training run) you want to evaluate. The script will automatically find a state dict of the required Generator and all the necessary information using `config.py`.

## Results

The model successfully learns to control the output audio's bandwidth according to the target sample rate.

The following image shows the full-spectrum mel-spectrograms of audio samples generated under some sampling rate condition (right-hand column) compared against mel-spectrograms of corresponding ground-truth audio (left-hand column). The red dashed line indicates the Nyquist frequency for each target rate, demonstrating the model's adherence to the physical constraints:

![Results](/pics/results.png)

## License

This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Acknowledgments

* This work is based on the original [HiFi-GAN](https://arxiv.org/abs/2010.05646) implementation and uses the principles from the [FiLM](https://arxiv.org/abs/1709.07871) paper for conditioning

* The project uses the [AudioSet](https://research.google.com/audioset/) dataset for training

* Thanks to [Prof Wenwu Wang](https://www.surrey.ac.uk/people/wenwu-wang) for his guidance and support