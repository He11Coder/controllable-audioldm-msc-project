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
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.native_sampling_rate, n_fft=config.n_fft, n_mels=config.n_mels,
            hop_length=config.hop_size, win_length=config.win_size,
            f_min=config.fmin, f_max=config.native_sampling_rate // 2, power=1.0
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filepath = self.audio_files[index]
        try:
            # Load audio file
            audio_native, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            # Return zero tensor on error
            return None #torch.zeros(self.config.segment_size)

        # Ensure mono channel
        if audio_native.shape[0] > 1:
            audio_native = torch.mean(audio_native, dim=0, keepdim=True)

        # Normalize audio
        audio_native = audio_native / (torch.max(torch.abs(audio_native)) + 1e-8)

        # Resample if necessary
        if sr != self.config.native_sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.native_sampling_rate)
            audio_native = resampler(audio_native)

        target_sr = random.choice(self.config.supported_sampling_rates)
        if target_sr != self.config.native_sampling_rate:
            resampler = torchaudio.transforms.Resample(self.config.native_sampling_rate, target_sr)
            audio_target = resampler(audio_native)
        else:
            audio_target = audio_native

        
        # Pad or slice to segment_size if adjust_to_seg_size=True
        if self.adjust_to_seg_size == True:
            if audio_target.size(1) >= self.config.segment_size:
                max_start_target = audio_target.size(1) - self.config.segment_size
                start_idx_target = random.randint(0, max_start_target)
                audio_segment_target = audio_target[:, start_idx_target : start_idx_target + self.config.segment_size]
            else:
                audio_segment_target = F.pad(audio_target, (0, self.config.segment_size - audio_target.size(1)), 'constant')

            start_idx_native = int(start_idx_target * (self.config.native_sampling_rate / target_sr))
            segment_size_native = int(self.config.segment_size * (self.config.native_sampling_rate / target_sr))

            if audio_native.size(1) < start_idx_native + segment_size_native:
                audio_native = F.pad(audio_native, (0, start_idx_native + segment_size_native - audio_native.size(1)))

            audio_segment_native = audio_native[:, start_idx_native : start_idx_native + segment_size_native]

        # Apply necessary MelSpectrogram transform
        mel_spec = self.mel_transform(audio_segment_native)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return {
            'audio': audio_segment_target.squeeze(0),
            'mel': log_mel_spec.squeeze(0),
            'sr': target_sr,
        }
    

def collate_fn(batch):
    """Custom collate function to filter out None values from failed loads."""
    batch = [b for b in batch if b is not None]

    if not batch:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)