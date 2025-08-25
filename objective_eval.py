import os
import torch
import torchaudio
import json
import random
import glob
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
from scipy.stats import entropy
from tqdm import tqdm

from models import Generator, SampleRateEmbedding
import config

class ObjEvaluator:
    """
    A class to handle the comprehensive objective evaluation of a trained conditional vocoder.
    """
    def __init__(self, config, checkpoint_path):
        """
        Initializes the evaluator.

        Args:
            config (HifiGanConfig): The configuration object used for training.
            checkpoint_path (str): Path to the saved generator checkpoint (.pt file).
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 1. Load your trained models
        self.embedding_layer = SampleRateEmbedding(config.sr_embedding_dim, config.supported_sampling_rates).to(self.device)
        self.generator = Generator(config).to(self.device)
        
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.generator.eval()
        print(f"Loaded generator from {checkpoint_path}")

        # 2. Load the model used for FAD and other metrics
        self.fad_model = FrechetAudioDistance(model_name="vggish", use_pca=False, use_activation=False, verbose=False)

    def generate_samples_for_eval(self, ground_truth_files, target_sr, output_dir):
        """
        Generates audio for evaluation and creates a corresponding set of
        real audio files at the target sample rate.
        """
        real_dir = os.path.join(output_dir, f"real_{target_sr}Hz")
        fake_dir = os.path.join(output_dir, f"fake_{target_sr}Hz")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

        print(f"Generating samples for {target_sr}Hz...")
        with torch.no_grad():
            for i, file_path in enumerate(tqdm(ground_truth_files, desc="Generating files")):
                # Load and prepare the ground truth audio
                audio_native, sr = torchaudio.load(file_path)
                if sr != self.config.native_sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.config.native_sampling_rate)
                    audio_native = resampler(audio_native)
                if audio_native.shape[0] > 1:
                    audio_native = torch.mean(audio_native, dim=0, keepdim=True)
                audio_native = audio_native.to(self.device)

                # Create the high-resolution mel spectrogram (the "content")
                mel_transform_native = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config.native_sampling_rate, n_fft=self.config.n_fft,
                    n_mels=self.config.n_mels, hop_length=self.config.hop_size,
                    win_length=self.config.win_size, f_min=self.config.fmin,
                    f_max=self.config.native_sampling_rate // 2, power=1.0
                ).to(self.device)
                input_mel = torch.log(torch.clamp(mel_transform_native(audio_native), min=1e-5))
                
                # Get the conditioning embedding for the target rate
                condition_embedding = self.embedding_layer(torch.tensor([target_sr])).to(self.device)

                # Generate the waveform at the native rate, conditioned on the target rate
                generated_waveform_native = self.generator(input_mel, condition_embedding)
                
                # Downsample the generated waveform to the target rate
                resampler_gen = torchaudio.transforms.Resample(self.config.native_sampling_rate, target_sr).to(self.device)
                generated_waveform_target = resampler_gen(generated_waveform_native)
                
                # Create the ground truth audio at the target rate
                resampler_gt = torchaudio.transforms.Resample(self.config.native_sampling_rate, target_sr).to(self.device)
                ground_truth_target = resampler_gt(audio_native)

                # Save both files
                torchaudio.save(os.path.join(fake_dir, f"{i}.wav"), generated_waveform_target.cpu(), target_sr)
                torchaudio.save(os.path.join(real_dir, f"{i}.wav"), ground_truth_target.cpu(), target_sr)

    def calculate_metrics(self, real_dir, fake_dir):
        """
        Calculates FAD, IS, and KL Divergence.
        """
        print("Calculating metrics...")
        # Calculate FAD score
        fad_score = self.fad_model.score(real_dir, fake_dir)

        # Get embeddings for IS and KL
        real_embeddings = self.fad_model.get_embeddings(real_dir)
        fake_embeddings = self.fad_model.get_embeddings(fake_dir)

        # Calculate KL Divergence
        # Create histograms (discrete probability distributions) from the embeddings
        real_hist, _ = np.histogram(real_embeddings.mean(axis=1), bins=50, density=True)
        fake_hist, _ = np.histogram(fake_embeddings.mean(axis=1), bins=50, density=True)

        # Add a small epsilon to avoid division by zero
        real_hist += 1e-10
        fake_hist += 1e-10
        kl_score = entropy(pk=fake_hist, qk=real_hist)

        # Calculate Inception Score (IS)
        # This is a simplified version for audio. It measures the KL divergence
        # of the marginal distribution from a uniform distribution.
        # A higher score indicates more confident and diverse predictions.
        preds = torch.nn.functional.softmax(torch.from_numpy(fake_embeddings), dim=1)
        marginal_dist = preds.mean(dim=0)
        is_scores = []
        for i in range(preds.size(0)):
            is_scores.append(entropy(preds[i], marginal_dist))
        is_score = np.exp(np.mean(is_scores))

        return {"fad": fad_score, "kl_divergence": kl_score, "inception_score": is_score}
    
    def run_full_evaluation(self, validation_files):
        """
        Orchestrates the entire evaluation process for all supported sample rates.
        """
        all_results = {}
        for sr in self.config.supported_sample_rates:
            print(f"\n--- Evaluating for {sr} Hz ---")
            output_dir = f"./evaluation/{sr}Hz"
            self.generate_samples_for_eval(validation_files, sr, output_dir)
            metrics = self.calculate_metrics(os.path.join(output_dir, f"real_{sr}Hz"), os.path.join(output_dir, f"fake_{sr}Hz"))
            all_results[sr] = metrics
            print(f"Results for {sr}Hz: {metrics}")
        
        # Save final results to a file
        results_path = "final_evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nFull evaluation complete. Results saved to {results_path}")


if __name__ == "__main__":
    config = config.HifiGanConfig()
    
    best_checkpoint_path = "/scratch/vb00479/audioset_balanced_22k/checkpoints/check.pt"
    
    # Get the list of validation files
    all_files = glob.glob(os.path.join(config.audio_path, '**', '*.wav'), recursive=True)
    random.seed(config.seed)
    random.shuffle(all_files)

    # Split
    split_idx = int(len(all_files) * config.validation_split_ratio)
    validation_files = all_files[:split_idx]

    num_eval_files = 50 
    validation_files = validation_files[:num_eval_files]
    
    if os.path.exists(best_checkpoint_path):
        evaluator = ObjEvaluator(config, best_checkpoint_path)
        evaluator.run_full_evaluation(validation_files)
    else:
        print(f"Error: Checkpoint file not found at {best_checkpoint_path}")