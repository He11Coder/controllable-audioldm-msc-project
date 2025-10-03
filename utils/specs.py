import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torchaudio

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
def mel_spectrogram(y, config, sampling_rate):
    """
    Converts a waveform tensor with a given sampling rate into a mel-spectrogram.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        hop_length=config.hop_size,
        win_length=config.win_size,
        f_min=config.fmin,
        f_max=sampling_rate // 2,
        power=1.0,
    ).to(y.device)

    mel_spec = mel_transform(y)

    # Convert to log-mel spectrogram
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    return log_mel_spec

# Helper function that applies MelTransform to the whole batch of audio clips
def mel_spec_for_batch(audio_batch, config, sr_batch):
    """Applies MelSpectrogram transform to each audio clip in `audio_batch` using corresponding sampling rates from `sr_batch`.
    If `sr_batch` is a single integer, in will be used as a sampling rate for each clip in `audio_batch`.

    Returns a batch (`torch.Tensor`) of the resulting spectrograms"""
    if isinstance(sr_batch, torch.Tensor):
        mel_list = []
        for audio, sr in zip(audio_batch, sr_batch):
            sr = sr.item()
            mel_list.append(mel_spectrogram(audio, config, sr))

        return torch.stack(mel_list)
    elif isinstance(sr_batch, int):
        return mel_spectrogram(audio_batch, config, sr_batch)

# Helper function for plotting linear spectrograms
def plot_linear_spectrogram_to_numpy(spectrogram, config, sampling_rate):
    """Converts a linear spectrogram tensor to a NumPy image array.
    The result contains Hz and time (sec) axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')

    # Convert Y-axis from bins to Hz
    num_freq_bins = spectrogram.shape[0]
    num_ticks_y = 8
    y_ticks = np.linspace(0, num_freq_bins - 1, num=num_ticks_y, dtype=np.intc)
    y_tick_labels = [f"{i * (sampling_rate / 2) / num_freq_bins / 1000:.1f}k" for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    # Convert X-axis from frames to time (in seconds)
    num_frames = spectrogram.shape[1]
    num_ticks_x = 5
    x_ticks = np.linspace(0, num_frames - 1, num=num_ticks_x, dtype=np.intc)
    x_tick_labels = [f"{i * config.hop_size / sampling_rate:.2f}" for i in x_ticks]
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
def plot_mel_spectrogram_to_numpy(spectrogram, config, sampling_rate):
    """Converts a mel-spectrogram tensor to a NumPy image array.
    The result contains Hz and time (sec) axes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    
    # Convert Y-axis from Mel bins to Hz
    mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=sampling_rate // 2)
    num_ticks_y = 8
    y_ticks = np.linspace(0, config.n_mels - 1, num=num_ticks_y, dtype=np.intc)
    y_tick_labels = [f"{mel_freqs[i]/1000:.1f}k" for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    
    # Convert X-axis from frames to time (in seconds)
    num_frames = spectrogram.shape[1]
    num_ticks_x = 5
    x_ticks = np.linspace(0, num_frames - 1, num=num_ticks_x, dtype=np.intc)
    x_tick_labels = [f"{i * config.hop_size / sampling_rate:.2f}" for i in x_ticks]
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