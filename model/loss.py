import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mel_processing import mel_spectrogram_torch
from utils.stft_loss import MultiResolutionSTFTLoss

    
class CVAEJETSLoss(nn.Module):
    def __init__(self, preprocess_config, model_config, train_config):
        super(CVAEJETSLoss, self).__init__()
        self.synthesizer_loss = SynthesizerLoss(preprocess_config, model_config, train_config)
        self.advdisc_loss = AdversarialDisciriminatorLoss()
        self.advgen_loss = AdversarialGeneratorLoss()
        self.fm_loss = FeatureMatchingLoss()
        
    def disc_loss_fn(self, disc_real_outputs, disc_generated_outputs):
        loss = self.advdisc_loss(disc_real_outputs, disc_generated_outputs)
        losses = {"loss/d/l_advdisc": loss}
        return loss, losses
    
    def gen_loss_fn(self, inputs, predictions, step, disc_outputs, fmap_r, fmap_g):
        l_synthe, losses = self.synthesizer_loss(inputs, predictions, step)
        l_advgen = self.advgen_loss(disc_outputs)
        l_fmatch = self.fm_loss(fmap_r, fmap_g)
        loss = l_synthe + l_advgen + l_fmatch
        losses.update({'loss/g/l_advgen':l_advgen,'loss/g/l_fmatch':l_fmatch}) 
        return loss, losses


class SynthesizerLoss(nn.Module):
    def __init__(self, preprocess_config, model_config, train_config):
        super(SynthesizerLoss, self).__init__()
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.binarization_loss_enable_steps = train_config['duration']['binarization_loss_enable_steps']
        self.binarization_loss_warmup_steps = train_config['duration']['binarization_loss_warmup_steps']
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            [1024, 2048, 512],
            [128, 256, 64],
            [1024, 2048, 512],
        )
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        # self.mse_loss = MSELoss()
        # self.mae_loss = MAELoss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.kld = KLDivergence()
        
    def get_mel(self, y):
        mel = mel_spectrogram_torch(
            y=y.squeeze(1), 
            n_fft=self.filter_length, 
            num_mels=self.n_mel_channels, 
            sampling_rate=self.sampling_rate, 
            hop_size=self.hop_length , 
            win_size=self.filter_length, 
            fmin=0, fmax=self.sampling_rate//2
        )[0]
        return mel

    def forward(self, inputs, predictions, step):
        (
            _, 
            _, 
            _, 
            _, 
            mel_targets,
            _,
            _,
            cwt_spec_targets,
            cwt_mean_targets,
            cwt_std_targets,
            uv_targets,
            energy_targets,
            _, 
            wav_targets, 
        ) = inputs
        
        (
            wav_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            duration_targets,
            src_masks,
            mel_masks, 
            indices, 
            src_lens,
            mel_lens,
            attn_hard, 
            attn_soft, 
            attn_logprob, 
            z_p, 
            m_p, 
            logs_p, 
            _, 
            logs_q
        ) = predictions
        
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, :, :mel_masks.shape[1]]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        cwt_spec_targets.requires_grad = False
        cwt_mean_targets.requires_grad = False
        cwt_std_targets.requires_grad = False
        uv_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        wav_targets.requires_grad = False
        
        (cwt_spec_predictions, cwt_mean_predictions, cwt_std_predictions, _) = pitch_predictions
        cwt_spec_predictions, uv_predictions = cwt_spec_predictions[:, :, :10], cwt_spec_predictions[:,:,-1]
        
        cwt_spec_loss = self.mse_loss(cwt_spec_predictions, cwt_spec_targets)
        cwt_mean_loss = self.mse_loss(cwt_mean_predictions, cwt_mean_targets)
        cwt_std_loss = self.mse_loss(cwt_std_predictions, cwt_std_targets)
        
        uv_std_loss = (
            F.binary_cross_entropy_with_logits(
                uv_predictions, uv_targets, reduction="none"
            ) * mel_masks.float()).sum() / mel_masks.float().sum()

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        
        wav_targets = wav_targets[:,indices[0]*self.hop_length:indices[1]*self.hop_length]
        wav_predictions = wav_predictions.squeeze(1)
        wav_targets = wav_targets.squeeze(1)
        assert wav_predictions.size() == wav_targets.size()
        stft_loss = self.stft_loss_fn(wav_predictions, wav_targets) * 20.
        
        # mel_predictions = self.get_mel(wav_predictions)[...,:indices[1]-indices[0]]
        # mel_targets = mel_targets[...,indices[0]:indices[1]]
        # assert mel_predictions.size() == mel_targets.size()
        # mel_loss = self.mae_loss(mel_predictions, mel_targets) * 45.

        ctc_loss = self.sum_loss(
            attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min(
                (step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
        
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        
        kl_loss = self.kld(z_p, m_p, logs_p, logs_q, mel_masks.unsqueeze(1))

        total_loss = (
            # mel_loss + 
            stft_loss + 
            duration_loss + 
            energy_loss + 
            ctc_loss + 
            bin_loss + 
            cwt_spec_loss + 
            cwt_mean_loss + 
            cwt_std_loss + 
            uv_std_loss + 
            kl_loss
        )
        
        losses = {
            "loss/g/bin_loss": bin_loss, 
            "loss/g/ctc_loss": ctc_loss, 
            "loss/g/duration": duration_loss,
            # "loss/g/mel": mel_loss,
            "loss/g/stft_loss": stft_loss,
            "loss/g/kl_loss": kl_loss, 
            "loss/g/energy": energy_loss,
            "loss/g/cwt_spec_loss": cwt_spec_loss, 
            "loss/g/cwt_mean_loss": cwt_mean_loss, 
            "loss/g/cwt_std_loss": cwt_std_loss, 
            "loss/g/uv_std_loss": uv_std_loss, 
        }

        return total_loss, losses

    
class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
    
    
class AdversarialDisciriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, disc_real_outputs, disc_generated_outputs):
        return discriminator_loss(disc_real_outputs, disc_generated_outputs)[0]

    
class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, disc_outputs):
        return generator_loss(disc_outputs)[0]
        
        
class FeatureMatchingLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, fmap_r, fmap_g):
        return feature_loss(fmap_r, fmap_g)
    
    
class KLDivergence(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, z_p, logs_q, m_p, logs_p, z_mask):
        return kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
    
    
class MSELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, prediction, target):
        return (prediction - target).pow(2).mean()
    
    
class MAELoss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, prediction, target):
        return (prediction - target).abs().mean()
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, m_p, logs_p, logs_q, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
