"""
This module contains an implementation of AudioDataset inherited from torch.Dataset.

It handles the raw audio data and provides necessary interfaces for retrieving it.
"""

import random
import math

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler
from torch.nn import functional as F

# Custom torch.Dataset ---
class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset to load audio files specified in `audio_files_list`.
    It loads audio clips, resamples them, and returns processed audio clips along with their mel-spectrograms and sampling rates.
    
    Random segments are returned if `adjust_to_seg_size=True`.

    See `HifiGanConfig` in `config.py` for more controllability.
    """
    def __init__(self, config, audio_files_list, adjust_to_seg_size=True, random_sampling_rate=True):
        super().__init__()
        self.config = config
        self.audio_files = audio_files_list
        self.adjust_to_seg_size = adjust_to_seg_size

        if random_sampling_rate:
            self.target_srs = [random.choice(self.config.supported_sampling_rates) for _ in range(len(self.audio_files))]
        else:
            self.target_srs = [self.config.supported_sampling_rates[index % len(self.config.supported_sampling_rates)] for index in range(len(self.audio_files))]
        #self.target_srs = []

        self.mel_transform_native = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.native_sampling_rate, n_fft=config.n_fft, n_mels=config.n_mels,
            hop_length=config.hop_size, win_length=config.win_size,
            f_min=config.fmin, f_max=config.native_sampling_rate // 2, power=1.0
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        filepath = self.audio_files[index]
        target_sr = self.target_srs[index]
        try:
            # Load audio file
            audio_native, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

        # Resample native version if necessary
        if sr != self.config.native_sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.native_sampling_rate)
            audio_native = resampler(audio_native)

        # Ensure mono channel
        if audio_native.shape[0] > 1:
            audio_native = torch.mean(audio_native, dim=0, keepdim=True)

        # Normalize audio
        audio_native = audio_native / (torch.max(torch.abs(audio_native)) + 1e-8)

        # Resample target version if necessary
        if target_sr != self.config.native_sampling_rate:
            resampler = torchaudio.transforms.Resample(self.config.native_sampling_rate, target_sr)
            audio_target = resampler(audio_native)
        else:
            audio_target = audio_native

        # Ensure mono channel
        if audio_target.shape[0] > 1:
            audio_target = torch.mean(audio_target, dim=0, keepdim=True)

        # Normalize audio
        audio_target = audio_target / (torch.max(torch.abs(audio_target)) + 1e-8)

        # Discard clip if it is too short
        if audio_target.size(1) < self.config.segment_size:
            return None

        audio_segment_target = audio_target
        audio_segment_native = audio_native
        if self.adjust_to_seg_size:
            max_start_target = audio_target.size(1) - self.config.segment_size
            start_idx_target = random.randint(0, max_start_target)
            audio_segment_target = audio_target[:, start_idx_target : start_idx_target + self.config.segment_size]

            start_idx_native = int(start_idx_target * (self.config.native_sampling_rate / target_sr))
            segment_size_native = int(self.config.segment_size * (self.config.native_sampling_rate / target_sr))
            audio_segment_native = audio_native[:, start_idx_native : start_idx_native + segment_size_native]

        # Apply MelSpectrogram transform to native segment
        mel_spec = self.mel_transform_native(audio_segment_native)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return {
            'audio': audio_segment_target.squeeze(0),
            'mel': log_mel_spec.squeeze(0),
            'sr': target_sr,
        }
    

class RateBucketingSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset.target_srs)

        # Group indices by their assigned sampling rate
        buckets = {sr: [] for sr in self.dataset.config.supported_sampling_rates}
        for i, sr in enumerate(self.dataset.target_srs):
            buckets[sr].append(i)

        # Create mini-batches for each sampling rate bucket
        batches = []
        for sr, indices in buckets.items():
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i : i+self.batch_size])
        
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return math.ceil(self.dataset.audio_files // self.batch_size)


def collate_fn(batch):
    """Custom collate function to filter out None values from failed loads."""
    batch = [b for b in batch if b is not None]

    if not batch:
        return None
    
    return torch.utils.data.dataloader.default_collate(batch)