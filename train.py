"""
Implementation of `Trainer` class which provides methods for model initialization, training, validation and logging.

Run this script to start training.
"""

import os
import glob
import itertools
import random
import time

import torch
import torch.optim as optim
import torchaudio
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F

import models
import audiodataset as ds
import utils
import config


class Trainer:
    """
    The main class to treat the HiFi-GAN model implemented in `models.py`. It is used for initialization, training and logging.
    """
    def __init__(self, config):
        self.conf = config
        torch.manual_seed(self.conf.seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup Logging
        self.sw = SummaryWriter(self.conf.tensorboard_path)

        # Setup Models
        self.embedding_layer = models.SampleRateEmbedding(self.conf.sr_embedding_dim, self.conf.supported_sampling_rates).to(self.device)
        self.generator = models.Generator(self.conf).to(self.device)
        self.mpd = models.MultiPeriodDiscriminator().to(self.device)
        self.msd = models.MultiScaleDiscriminator().to(self.device)
        
        # Setup Optimizers and Schedulers
        self.optim_g = optim.AdamW(self.generator.parameters(), self.conf.learning_rate_g, betas=[self.conf.adam_b1, self.conf.adam_b2])
        self.optim_d = optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()), self.conf.learning_rate_d, betas=[self.conf.adam_b1, self.conf.adam_b2])

        self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.conf.lr_decay)
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.conf.lr_decay)

        # Setup DataLoaders
        self._setup_dataloaders()

        os.makedirs(self.conf.checkpoint_path, exist_ok=True)
        
    def _setup_dataloaders(self):
        audio_files = glob.glob(os.path.join(self.conf.audio_path, '**', '*.wav'), recursive=True)
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.conf.audio_path}.")
        
        print(f"Found {len(audio_files)} audio files.")

        audio_files_to_vis = glob.glob(os.path.join(self.conf.audio_to_visualize_path, '*.wav'))
        if not audio_files_to_vis:
            raise FileNotFoundError(f"No audio files found in {self.conf.audio_to_visualize_path}.")
        
        print(f"Found {len(audio_files_to_vis)} audio files to visualize the training progess")

        random.seed(self.conf.seed)
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * self.conf.validation_split_ratio)
        val_files = audio_files[:split_idx]
        train_files = audio_files[split_idx:]

        train_dataset = ds.AudioDataset(self.conf, train_files)
        val_dataset = ds.AudioDataset(self.conf, val_files)
        visual_dataset = ds.AudioDataset(self.conf, audio_files_to_vis, adjust_to_seg_size=False, random_sampling_rate=False)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.conf.batch_size, shuffle=True, collate_fn=ds.collate_fn, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.conf.batch_size, shuffle=False, collate_fn=ds.collate_fn, num_workers=4, pin_memory=True)
        self.visual_loader = DataLoader(visual_dataset, batch_size=1, shuffle=False, collate_fn=ds.collate_fn, num_workers=4, pin_memory=True)

    def _mel_spec_for_batch(self, audio_batch, sr_batch):
        mel_list = []
        for audio, sr in zip(audio_batch, sr_batch):
            sr = sr.item()
            mel_list.append(utils.mel_spectrogram(audio, self.conf, sr))
        
        return torch.stack(mel_list)

    def _log_training(self, step, start_time, loss_gen, loss_disc, loss_mel, high_freq_loss):
        self.sw.add_scalar("Loss/Train/Generator_total", loss_gen.item(), step)
        self.sw.add_scalar("Loss/Train/Discriminator_total", loss_disc.item(), step)
        self.sw.add_scalar("Loss/Train/Mel_Spectrogram", loss_mel.item(), step)
        self.sw.add_scalar("Loss/Train/High_Freq_Loss", loss_mel.item(), step)

        elapsed_time = time.time() - start_time

        log_str = (f"Step: {step}, Gen Loss: {loss_gen.item():.4f}, "
                   f"Disc Loss: {loss_disc.item():.4f}, Mel Loss: {loss_mel.item():.4f}, "
                   f"High Freq Loss: {high_freq_loss.item():.4f}, "
                   f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        print(log_str)

    def _validate(self, step, start_time):
        self.generator.eval()
        self.mpd.eval()
        self.msd.eval()
        
        total_gen_loss, total_disc_loss, total_mel_loss, total_high_freq_loss = 0, 0, 0, 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= self.conf.num_batches_to_val:
                    break

                if batch is None:
                    continue

                y, mel, sr = batch['audio'], batch['mel'], batch['sr']

                y, mel, sr = y.to(self.device).unsqueeze(1), mel.to(self.device), sr.to(self.device)
                condition = self.embedding_layer(sr)

                y_g_hat = self.generator(mel, condition)
                #y_g_hat = y_g_hat[:, :, :self.conf.segment_size]
                segment_size_native = int(self.conf.segment_size * (self.conf.native_sampling_rate / sr[0].item()))
                y_g_hat = y_g_hat[:, :, :segment_size_native]

                y_upsampled = []
                for i in range(y.size(0)):
                    resampler_native = torchaudio.transforms.Resample(sr[i].item(), self.conf.native_sampling_rate).to(self.device)
                    y_native = resampler_native(y[i])

                    min_len = min(y_native.size(1), y_g_hat[i].size(1))
                    y_native = y_native[:, :min_len]

                    y_upsampled.append(y_native)

                y_upsampled = torch.stack(y_upsampled)

                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y_upsampled, y_g_hat)
                loss_disc_f = models.discriminator_loss(y_df_hat_r, y_df_hat_g)

                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y_upsampled, y_g_hat)
                loss_disc_s = models.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                total_disc_loss += (loss_disc_s + loss_disc_f).item()

                loss_mel = 0
                mel_transform_native = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.conf.native_sampling_rate, n_fft=self.conf.n_fft, n_mels=self.conf.n_mels,
                    hop_length=self.conf.hop_size, win_length=self.conf.win_size,
                    f_min=self.conf.fmin, f_max=self.conf.native_sampling_rate // 2, power=1.0
                ).to(self.device)

                for i in range(y.size(0)):
                    resampler_native = torchaudio.transforms.Resample(sr[i].item(), self.conf.native_sampling_rate).to(self.device)
                    y_native = resampler_native(y[i])

                    min_len = min(y_native.size(1), y_g_hat[i].size(1))
                    y_native = y_native[:, :min_len]
                    y_g_hat_i = y_g_hat[i][:, :min_len]

                    mel_g_hat_target = mel_transform_native(y_g_hat_i)
                    mel_target = mel_transform_native(y_native)
                    loss_mel += F.l1_loss(mel_g_hat_target, mel_target)

                loss_mel /= y.size(0)

                #mel_g_hat = self._mel_spec_for_batch(y_g_hat.squeeze(1), sr)
                #loss_mel = F.l1_loss(mel, mel_g_hat)

                _, _, fmap_f_r, fmap_f_g = self.mpd(y_upsampled, y_g_hat)
                _, _, fmap_s_r, fmap_s_g = self.msd(y_upsampled, y_g_hat)

                loss_fm_f = models.feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = models.feature_loss(fmap_s_r, fmap_s_g)

                loss_gen_f = models.generator_loss(y_df_hat_g)
                loss_gen_s = models.generator_loss(y_ds_hat_g)

                high_freq_loss = models.spectral_cutoff_loss(y_g_hat, sr, self.conf)

                total_gen_loss += (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + 45*loss_mel +10*high_freq_loss).item()
                total_mel_loss += loss_mel.item()
                total_high_freq_loss += high_freq_loss.item()


        avg_gen_loss = total_gen_loss / (i + 1)
        avg_disc_loss = total_disc_loss / (i + 1)
        avg_mel_loss = total_mel_loss / (i + 1)
        avg_high_freq_loss = total_high_freq_loss / (i + 1)

        self.sw.add_scalar("Loss/Validation/Generator_total", avg_gen_loss, step)
        self.sw.add_scalar("Loss/Validation/Discriminator_total", avg_disc_loss, step)
        self.sw.add_scalar("Loss/Validation/Mel_Spectrogram", avg_mel_loss, step)
        self.sw.add_scalar("Loss/Validation/High_Freq_Loss", avg_high_freq_loss, step)

        elapsed_time = time.time() - start_time

        log_str = (f"Validation - Gen Loss: {avg_gen_loss:.4f}, "
                   f"Disc Loss: {avg_disc_loss:.4f}, "
                   f"Mel Loss: {avg_mel_loss:.4f}, "
                   f"High Freq Loss: {avg_high_freq_loss:.4f}, "
                   f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        print(log_str)

        self.generator.train()
        self.mpd.train()
        self.msd.train()

        return avg_gen_loss, avg_disc_loss, avg_mel_loss, avg_high_freq_loss

    def _visualize_training(self, step):
        self.generator.eval()

        with torch.no_grad():
            sample_counter = 0
            for batch in self.visual_loader:
                if batch is None:
                    continue

                y, mel, sr = batch['audio'], batch['mel'], batch['sr']

                y, mel, sr = y.to(self.device).unsqueeze(1), mel.to(self.device), sr.to(self.device)
                condition = self.embedding_layer(sr)

                y_g_hat = self.generator(mel, condition)

                for real_audio, gen_audio, samp_rate in zip(y, y_g_hat, sr):
                    if step == 0:
                        self.sw.add_audio(f'Audio/Real_{sample_counter+1}', snd_tensor=real_audio, sample_rate=samp_rate.item())

                        real_mel_spec = utils.mel_spectrogram(real_audio, self.conf, samp_rate.item())
                        self.sw.add_image(f'Mel_Spectrogram/Real_{sample_counter+1}', img_tensor=utils.plot_mel_spectrogram_to_numpy(real_mel_spec.squeeze(0).cpu().detach().numpy(), self.conf, samp_rate.item()), dataformats='HWC')

                        real_linear_spec = utils.linear_spectrogram(real_audio, self.conf)
                        self.sw.add_image(f'Linear_Spectrogram/Real_{sample_counter+1}', img_tensor=utils.plot_linear_spectrogram_to_numpy(real_linear_spec.squeeze(0).cpu().detach().numpy(), self.conf, samp_rate.item()), dataformats='HWC')

                    self.sw.add_audio(f'Audio/Generated _{sample_counter+1}_{samp_rate}Hz', snd_tensor=gen_audio, global_step=step, sample_rate=self.conf.native_sampling_rate)
                            
                    gen_mel_spec = utils.mel_spectrogram(gen_audio, self.conf, self.conf.native_sampling_rate)
                    self.sw.add_image(f'Mel_Spectrogram/Generated_{sample_counter+1}_{samp_rate}Hz', img_tensor=utils.plot_mel_spectrogram_to_numpy(gen_mel_spec.squeeze(0).cpu().detach().numpy(), self.conf, self.conf.native_sampling_rate), global_step=step, dataformats='HWC')

                    gen_linear_spec = utils.linear_spectrogram(gen_audio, self.conf)
                    self.sw.add_image(f'Linear_Spectrogram/Generated_{sample_counter+1}_{samp_rate}Hz', img_tensor=utils.plot_linear_spectrogram_to_numpy(gen_linear_spec.squeeze(0).cpu().detach().numpy(), self.conf, self.conf.native_sampling_rate), global_step=step, dataformats='HWC')
                    
                    sample_counter += 1

        self.generator.train()

    def train(self):
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        step = 0
        data_iterator = iter(self.train_loader)
        start_time = time.time()
        
        print("Starting training...")

        while step <= self.conf.steps:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_loader)
                batch = next(data_iterator)

            if batch is None:
                continue

            y, mel, sr = batch['audio'], batch['mel'], batch['sr']

            y, mel, sr = y.to(self.device).unsqueeze(1), mel.to(self.device), sr.to(self.device)
            condition = self.embedding_layer(sr)

            y_g_hat = self.generator(mel, condition)
            segment_size_native = int(self.conf.segment_size * (self.conf.native_sampling_rate / sr[0].item()))
            y_g_hat = y_g_hat[:, :, :segment_size_native]

            y_upsampled = []
            for i in range(y.size(0)):
                resampler_native = torchaudio.transforms.Resample(sr[i].item(), self.conf.native_sampling_rate).to(self.device)
                y_native = resampler_native(y[i])

                min_len = min(y_native.size(1), y_g_hat[i].size(1))
                y_native = y_native[:, :min_len]

                y_upsampled.append(y_native)

            y_upsampled = torch.stack(y_upsampled)

            # Discriminator Training
            self.optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y_upsampled, y_g_hat.detach())
            loss_disc_f = models.discriminator_loss(y_df_hat_r, y_df_hat_g)

            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y_upsampled, y_g_hat.detach())
            loss_disc_s = models.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            self.optim_d.step()

            # Generator Training
            self.optim_g.zero_grad()

            loss_mel = 0
            mel_transform_native = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.conf.native_sampling_rate, n_fft=self.conf.n_fft, n_mels=self.conf.n_mels,
                hop_length=self.conf.hop_size, win_length=self.conf.win_size,
                f_min=self.conf.fmin, f_max=self.conf.native_sampling_rate // 2, power=1.0
            ).to(self.device)

            for i in range(y.size(0)):
                resampler_native = torchaudio.transforms.Resample(sr[i].item(), self.conf.native_sampling_rate).to(self.device)
                y_native = resampler_native(y[i])

                min_len = min(y_native.size(1), y_g_hat[i].size(1))
                y_native = y_native[:, :min_len]
                y_g_hat_i = y_g_hat[i][:, :min_len]

                mel_g_hat_target = mel_transform_native(y_g_hat_i)
                mel_target = mel_transform_native(y_native)
                loss_mel += F.l1_loss(mel_g_hat_target, mel_target)

            loss_mel /= y.size(0)

            #mel_g_hat = self._mel_spec_for_batch(y_g_hat.squeeze(1), sr)
            #loss_mel = F.l1_loss(mel, mel_g_hat)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y_upsampled, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y_upsampled, y_g_hat)

            loss_fm_f = models.feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = models.feature_loss(fmap_s_r, fmap_s_g)

            loss_gen_f = models.generator_loss(y_df_hat_g)
            loss_gen_s = models.generator_loss(y_ds_hat_g)

            high_freq_loss = models.spectral_cutoff_loss(y_g_hat, sr, self.conf)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + 45*loss_mel + 100*high_freq_loss
            loss_gen_all.backward()
            self.optim_g.step()

            # Logging and Checkpointing
            if step % self.conf.log_interval == 0:
                self._log_training(step, start_time, loss_gen_all, loss_disc_all, loss_mel, high_freq_loss)

            if step % self.conf.validation_interval == 0:
                val_gen_loss, val_disc_loss, val_mel_loss, val_high_freq_loss = self._validate(step, start_time)

            if step % self.conf.visualization_interval == 0 or step == 1:
                self._visualize_training(step)
            
            if step % self.conf.checkpoint_interval == 0 and step != 0:
                checkpoint_g_path = os.path.join(self.conf.checkpoint_path, f"g_{step:08d}.pt")
                torch.save(self.generator.state_dict(), checkpoint_g_path)

                elapsed_time = time.time() - start_time
                print(f"Saved checkpoint at step {step}, "
                      f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

                self.scheduler_g.step()
                self.scheduler_d.step()

            step += 1
        
        # Fix the final validation metrics
        final_metric_dict = {
            "Final_loss/Generator": val_gen_loss,
            "Final_loss/Discriminator": val_disc_loss,
            "Final_loss/Mel_Spectrogram": val_mel_loss,
            "Final_loss/High_Freq_Loss": val_high_freq_loss,
        }
        # Associate the metrics with the hyperparameters and log
        self.sw.add_hparams(config.hparam_dict, final_metric_dict)

        self.sw.close()


if __name__ == "__main__":
    conf = config.HifiGanConfig()

    trainer = Trainer(conf)
    trainer.train()