import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torchaudio

# Random (Gaussian) weights initialization
def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize model with Gaussian weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

# Linear Spectrogram Conversion
def linear_spectrogram(y, config):
    """
    Converts a waveform tensor into a linear spectrogram.
    """
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=config.n_fft, hop_length=config.hop_size, win_length=config.win_size,
        power=1.0
    ).to(y.device)

    spec = spec_transform(y)
    log_spec = torch.log(torch.clamp(spec, min=1e-5))
    return log_spec

# Mel-Spectrogram Conversion
def mel_spectrogram(y, config):
    """
    Converts a waveform tensor into a mel-spectrogram.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sampling_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        hop_length=config.hop_size,
        win_length=config.win_size,
        f_min=config.fmin,
        f_max=config.fmax,
        power=1.0,
    ).to(y.device)

    mel_spec = mel_transform(y)

    # Convert to log-mel spectrogram
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel_spec

# Helper function for plotting linear spectrograms
def plot_linear_spectrogram_to_numpy(spectrogram, config):
    """Converts a linear spectrogram tensor to a NumPy image array.
    The result contains Hz and time (sec) axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')

    # Convert Y-axis from bins to Hz
    num_freq_bins = spectrogram.shape[0]
    num_ticks_y = 8
    y_ticks = np.linspace(0, num_freq_bins - 1, num=num_ticks_y, dtype=np.intc)
    y_tick_labels = [f"{i * (config.sampling_rate / 2) / num_freq_bins / 1000:.1f}k" for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    # Convert X-axis from frames to time (in seconds)
    num_frames = spectrogram.shape[1]
    num_ticks_x = 5
    x_ticks = np.linspace(0, num_frames - 1, num=num_ticks_x, dtype=np.intc)
    x_tick_labels = [f"{i * config.hop_size / config.sampling_rate:.2f}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    plt.colorbar(im, ax=ax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    fig.canvas.draw()
    
    # Convert the plot to a NumPy array
    img_arr =np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_arr

# Helper function for plotting mel-spectrograms
def plot_mel_spectrogram_to_numpy(spectrogram, config):
    """Converts a mel-spectrogram tensor to a NumPy image array.
    The result contains Hz and time (sec) axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    
    # Convert Y-axis from Mel bins to Hz
    mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax)
    num_ticks_y = 8
    y_ticks = np.linspace(0, config.n_mels - 1, num=num_ticks_y, dtype=np.intc)
    y_tick_labels = [f"{mel_freqs[i]/1000:.1f}k" for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    # Convert X-axis from frames to time (in seconds)
    num_frames = spectrogram.shape[1]
    num_ticks_x = 5
    x_ticks = np.linspace(0, num_frames - 1, num=num_ticks_x, dtype=np.intc)
    x_tick_labels = [f"{i * config.hop_size / config.sampling_rate:.2f}" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    plt.colorbar(im, ax=ax)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    fig.canvas.draw()
    
    # Convert the plot to a NumPy array
    img_arr =np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_arr