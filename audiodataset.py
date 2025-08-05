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
    It loads audio clips, resamples them, and returns. Random segments are returned if `adjust_to_seg_size=True`.

    See `HifiGanConfig` in `config.py` for more controllability.
    """
    def __init__(self, config, audio_files_list, adjust_to_seg_size=True):
        super().__init__()
        self.config = config
        self.audio_files = audio_files_list
        self.adjust_to_seg_size = adjust_to_seg_size

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
            return torch.zeros(self.config.segment_size)

        # Ensure mono channel
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.config.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sampling_rate)
            audio = resampler(audio)

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

        return audio.squeeze(0)