import argparse
import os

import torch
import yaml
import torch.nn as nn

from utils.model import get_model
from utils.tools import to_device, to_device_inference, log, plot_spectrogram_to_numpy, plot_alignment_to_numpy
from model import CVAEJETSLoss
from data_utils import AudioTextDataset, AudioTextCollate, DataLoader
from mel_processing import mel_spectrogram_torch


def evaluate(models, step, configs, device, logger=None):
    model, discriminator = models
    preprocess_config, model_config, train_config = configs
    hop_size = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]

    # Get dataset
    dataset = AudioTextDataset(
        preprocess_config['path']['validation_files'], preprocess_config)
    
    batch_size = train_config["optimizer"]["batch_size"]
    collate_fn = AudioTextCollate()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=8, 
        pin_memory=True, 
        drop_last=False
    )

    # Get loss function
    Loss = CVAEJETSLoss(preprocess_config, model_config, train_config).to(device)

    # Evaluation
    loss_sums_disc = [0 for _ in range(1)] # + total
    loss_sums_model = [0 for _ in range(12)] # + total
    for batch in loader:
        batch = to_device(batch, device)
    
        with torch.no_grad():
            output = model(*(batch[:-1]), step=step, gen=False)
            
            wav_predictions, indices = output[0], output[7]
            wav_targets = batch[-1][...,indices[0]*hop_size:indices[1]*hop_size]
            
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(wav_targets.unsqueeze(1), wav_predictions)

            loss_disc, losses_disc = Loss.disc_loss_fn(
                disc_real_outputs=y_d_hat_r, 
                disc_generated_outputs=y_d_hat_g)
            
            loss_model, losses_model = Loss.gen_loss_fn(
                inputs=batch, 
                predictions=output, 
                step=step, 
                disc_outputs=y_d_hat_g, 
                fmap_r=fmap_r, 
                fmap_g=fmap_g)

            for i in range(len(losses_disc)):
                loss_sums_disc[i] += list(losses_disc.values())[i].item() * len(batch[0])
            for i in range(len(losses_model)):
                loss_sums_model[i] += list(losses_model.values())[i].item() * len(batch[0])
    
    # get scalars
    loss_means_disc = [loss_sum / len(dataset) for loss_sum in loss_sums_disc]
    loss_means_model = [loss_sum / len(dataset) for loss_sum in loss_sums_model]
    scalars_disc = {k:v for k,v in zip(losses_disc.keys(), loss_means_disc)}
    scalars_model = {k:v for k,v in zip(losses_model.keys(), loss_means_model)}

    message1 = f"Discriminator Validation Step {step}, " + " ".join([str(round(l.item(), 4)) for l in losses_disc.values()]).strip()
    message2 = f"Model Validation Step {step}, " + " ".join([str(round(l.item(), 4)) for l in losses_model.values()]).strip()
    message = f"{message1}\n{message2}"
    
    # synthesis one sample
    with torch.no_grad():
        # segmented output
        for i in range(len(batch)-1):
            try:
                batch[i] = batch[i][:1]
            except:
                pass
            
        output = model(*(batch[:-1]), step=step)
        wav = output[0]
        mel = Loss.synthesizer_loss.get_mel(wav)
        wav_len = output[9][0].item() * hop_size
        attn_h = output[10]
        attn_s = output[11]
        
        # total output
        pairs = to_device_inference(
            [batch[0][:1], batch[1][:1], batch[2][:1], None], device)
        output_gen = model(*(pairs), gen=True)
        wav_gen = output_gen[0]
        mel_gen = Loss.synthesizer_loss.get_mel(wav_gen)
        wav_gen_len = output_gen[9][0].item() * hop_size
    
    image_dict = {
        "gen/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy()), 
        "gen/mel_gen": plot_spectrogram_to_numpy(mel_gen[0].cpu().numpy()), 
        "all/attn_h": plot_alignment_to_numpy(attn_h[0,0].data.cpu().numpy()), 
        "all/attn_s": plot_alignment_to_numpy(attn_s[0,0].data.cpu().numpy())
    }
    audio_dict = {
      "gen/audio": wav[0,:,:wav_len], 
      "gen/audio_gen": wav_gen[0,:,:wav_gen_len]
    }
    scalar_dict = {}
    scalar_dict.update(scalars_disc)
    scalar_dict.update(scalars_model)
    if logger is not None:
        log(writer=logger,
            global_step=step, 
            images=image_dict,
            audios=audio_dict, 
            scalars=scalar_dict, 
            audio_sampling_rate=sampling_rate)

    return message
    