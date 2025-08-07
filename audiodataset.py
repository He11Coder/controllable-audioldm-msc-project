"""
This module contains an implementation of AudioDataset inherited from torch.Dataset.

It handles the raw audio data and provides necessary interfaces for retrieving it.
"""

import random

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn import functional as F

# Custom torch.Dataset ---
class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset to load audio files specified in `audio_files_list`.
    It loads audio clips, resamples them, and returns processed audio clips along with their mel-spectrograms and sampling rates.
    
    Random segments are returned if `adjust_to_seg_size=True`.

    See `HifiGanConfig` in `config.py` for more controllability.
    """
    def __init__(self, config, audio_files_list, adjust_to_seg_size=True):
        super().__init__()
        self.config = config
        self.audio_files = audio_files_list
        self.adjust_to_seg_size = adjust_to_seg_size
        self.mel_transforms = {
            sr: torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=config.n_fft, n_mels=config.n_mels,
                hop_length=config.hop_size, win_length=config.win_size,
                f_min=config.fmin, f_max=sr // 2, power=1.0
            ) for sr in config.supported_sampling_rates
        }

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filepath = self.audio_files[index]
        try:
            # Load audio file
            audio, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            # Return zero tensor on error
            return None #torch.zeros(self.config.segment_size)

        # Resample if necessary
        target_sr = random.choice(self.config.supported_sampling_rates)
        if target_sr != sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # Ensure mono channel
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Normalize audio
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        
        # Pad or slice to segment_size if adjust_to_seg_size=True
        if self.adjust_to_seg_size == True:
            if audio.size(1) >= self.config.segment_size:
                max_start = audio.size(1) - self.config.segment_size
                start_idx = random.randint(0, max_start)
                audio = audio[:, start_idx : start_idx + self.config.segment_size]
            else:
                audio = F.pad(audio, (0, self.config.segment_size - audio.size(1)), 'constant')

        # Apply necessary MelSpectrogram transform
        mel_transform = self.mel_transforms[target_sr]
        mel_spec = mel_transform(audio)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return {
            'audio': audio.squeeze(0),
            'mel': log_mel_spec.squeeze(0),
            'sr': target_sr,
        }
    

def collate_fn(batch):
    """Custom collate function to filter out None values from failed loads."""
    batch = [b for b in batch if b is not None]

    if not batch:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)