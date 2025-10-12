import torch

def spectral_cutoff_loss(y_g_hat, target_sr_batch, config):
    """
    Calculates a loss based on spectral energy above the Nyquist frequency.

    This function provides a direct, physically-grounded penalty to the generator
    for producing frequencies that should not exist in an audio signal of a given
    target sample rate. It works by:
    1. Performing a Short-Time Fourier Transform (STFT) on the generated high-rate audio.
    2. For each sample in the batch, identifying the frequency bin corresponding to its
       target sample rate's Nyquist frequency.
    3. Calculating the mean magnitude of all frequency content *above* this cutoff bin.
    4. Averaging this penalty across the batch to form the final loss.

    Args:
        y_g_hat (torch.Tensor): The batch of generated audio waveforms from the generator,
            at the native high sampling rate. Shape: (batch_size, 1, length).
        target_sr_batch (torch.Tensor): A batch of target sample rates corresponding
            to each waveform in `y_g_hat`. Shape: (batch_size,).
        config (HifiGanConfig): The configuration object containing STFT parameters.

    Returns:
        torch.Tensor: A scalar tensor representing the average spectral cutoff loss for the batch.
    """
    window = torch.hann_window(config.win_size).to(y_g_hat.device)

    stft = torch.stft(y_g_hat.squeeze(1), n_fft=config.n_fft, hop_length=config.hop_size, 
                      win_length=config.win_size, window=window, return_complex=True)
    magnitudes = torch.abs(stft)

    loss = torch.tensor(0.0, device=y_g_hat.device, requires_grad=True)
    for i in range(magnitudes.size(0)):
        target_sr = target_sr_batch[i].item()
        if target_sr >= config.native_sampling_rate:
            continue

        nyquist_freq = target_sr / 2
        freq_bin_width = config.native_sampling_rate / config.n_fft
        cutoff_bin = int(nyquist_freq / freq_bin_width)

        undesired_energy = magnitudes[i, cutoff_bin:, :]
        loss = loss + torch.mean(undesired_energy)
    
    return loss / magnitudes.size(0)