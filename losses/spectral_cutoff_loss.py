import torch

def spectral_cutoff_loss(y_g_hat, target_sr_batch, config):
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