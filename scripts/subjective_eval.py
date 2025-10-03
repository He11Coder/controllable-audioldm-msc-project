import os
import glob
import random
import matplotlib.pyplot as plt

import torch
import torchaudio
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from utils.init_weights import mel_spectrogram, linear_spectrogram
from models import SampleRateEmbedding, Generator


class EvalDataset(IterableDataset):
    def __init__(self, config, audio_files_list, sampling_rates_to_eval):
        super().__init__()
        self.conf = config
        self.audio_files = audio_files_list
        self.sampling_rates = sampling_rates_to_eval
    
    def __iter__(self):
        #random.shuffle(self.audio_files)
        for i, file in enumerate(self.audio_files):
            audio, sr = torchaudio.load(file)

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)

            for target_sr in self.sampling_rates:
                if sr != self.conf.native_sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.conf.native_sampling_rate)
                    audio = resampler(audio)
                
                mel = mel_spectrogram(audio, self.conf, self.conf.native_sampling_rate)

                yield {
                    'file_num': i+1,
                    'audio': audio,
                    'mel': mel,
                    'sr': target_sr,
                }


class Evaluator:
    def __init__(self, config, sampling_rates_to_eval):
        self.conf = config
        self.sampling_rates_to_eval = sampling_rates_to_eval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for evaluation: {self.device}")
        
        # TODO: implement saving media to TensorBoard
        #self.sw = SummaryWriter(os.path.join("hifi-gan_eval_runs", self.conf.run_name))
        os.makedirs(os.path.join("hifi-gan_eval_runs", self.conf.run_name), exist_ok=True)

        self.embedding_layer = SampleRateEmbedding(self.conf.sr_embedding_dim, self.conf.supported_sampling_rates).to(self.device)
        self.generator = Generator(self.conf).to(self.device)

        checkpoint_file_path = os.path.join(self.conf.checkpoint_path, f"g_{self.conf.steps:08d}.pt")
        self.generator.load_state_dict(torch.load(checkpoint_file_path, map_location=self.device))
        self.generator.eval()

        self._setup_dataloaders()

    def _setup_dataloaders(self):
        audio_files = glob.glob(os.path.join(self.conf.audio_to_visualize_path, '**', '*.wav'), recursive=True)
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.conf.audio_path}.")
        
        print(f"Found {len(audio_files)} audio files.")

        dataset = EvalDataset(self.conf, audio_files, self.sampling_rates_to_eval)
        self.dataloader = DataLoader(dataset, batch_size=None, pin_memory=True)

    def _get_interpolated_embedding(self, target_sr):
        supported_rates = sorted(self.conf.supported_sampling_rates)

        # Find the two closest anchor rates (neighbours)
        if target_sr <= supported_rates[0]:
            return self.embedding_layer(torch.tensor([supported_rates[0]]))
        if target_sr >= supported_rates[-1]:
            return self.embedding_layer(torch.tensor([supported_rates[-1]]))
        
        for i in range(len(supported_rates)-1):
            sr_low, sr_high = supported_rates[i], supported_rates[i+1]
            if sr_low <= target_sr <= sr_high:
                break

        # Get the anchor embeddings
        emb_low = self.embedding_layer(torch.tensor([sr_low]))
        emb_high = self.embedding_layer(torch.tensor([sr_high]))

        # Calculate the linear interpolation weight
        w = (target_sr - sr_low) / (sr_high - sr_low)

        # Perform the interpolation
        interpolated_emb = (1 - w) * emb_low + w * emb_high

        return interpolated_emb

    def evaluate_interpolation(self):
        for i, data in enumerate(self.dataloader):
            audio = data['audio']
            mel = data['mel'].to(self.device)
            target_sr = data['sr']

            condition_embedding = self._get_interpolated_embedding(target_sr).to(self.device)

            with torch.no_grad():
                gen_audio = self.generator(mel, condition_embedding)
            
            # TODO: SAVE THE OUTPUT AUDIO

            # Create the ground-truth version at the interpolated rate
            #gt_resampler = torchaudio.transforms.Resample(48000, 96000)
            #ground_truth_interpolated = gt_resampler(audio)

            '''gen_tensor = linear_spectrogram(gen_audio, self.conf)
            ps(gen_tensor.squeeze(0), os.path.join(os.path.join("hifi-gan_eval_runs", self.conf.run_name), f"{i+1}_gen_spectrogram_{target_sr}Hz.png"))

            real_tensor = linear_spectrogram(audio, self.conf)
            ps(real_tensor, os.path.join(os.path.join("hifi-gan_eval_runs", self.conf.run_name), f"{i+1}_real_spectrogram_{target_sr}Hz.png"))'''

            # --- Visual Verification of Frequency Content ---
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))

            # Plot ground truth
            #plot_spectrogram(ground_truth_interpolated.cpu(), 96000, "Ground Truth Spectrogram", ax1)

            #gt_resampler = torchaudio.transforms.Resample(target_sr, 96000)
            #gen_interpolated = gt_resampler(gen_audio.cpu())

            # Plot generated

            plot_spectrogram(gen_audio.cpu(), target_sr, "Generated Spectrogram", ax1)

            # Add a line at the Nyquist frequency for reference
            nyquist_freq_khz = (target_sr / 2) / 1000
            ax1.axhline(y=nyquist_freq_khz, color='r', linestyle='--', label=f'Nyquist Freq ({nyquist_freq_khz} kHz)')
            #ax2.axhline(y=nyquist_freq_khz, color='r', linestyle='--')
            ax1.legend()

            plt.tight_layout()
            plot_path = os.path.join("hifi-gan_eval_runs", self.conf.run_name, f"{data['file_num']}", f"{i+1}_spectrogram_comparison_{target_sr}Hz.png")
            os.makedirs(os.path.join("hifi-gan_eval_runs", self.conf.run_name, f"{data['file_num']}"), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Saved spectrogram comparison plot to {plot_path}")
            plt.close(fig)


def plot_spectrogram(waveform, sr, title, ax):
    """Helper function to plot a linear spectrogram."""
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=256,
        win_length=1024
    )
    spec = spectrogram_transform(waveform)
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    
    im = ax.imshow(spec_db[0].squeeze(0).numpy(), aspect='auto', origin='lower',
                   extent=[0, waveform.shape[1] / sr, 0, sr / 2 / 1000])
    ax.set_title(title)
    ax.set_ylabel("Frequency (kHz)")
    ax.set_xlabel("Time (s)")
    return im

if __name__ == "__main__":
    conf = config.HifiGanConfig()
    
    evaluator = Evaluator(conf, conf.supported_sampling_rates)
    evaluator.evaluate_interpolation()