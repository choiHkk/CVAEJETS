import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from scipy.stats import betabinom
from tqdm.auto import tqdm
import numpy as np
import librosa
import random
import torch
import json
import os

from utils import tools, pitch_utils
from text import text_to_sequence
# import audio as Audio
from mel_processing import mel_spectrogram_torch
random.seed(1234)



class AudioTextCollate(object):
    def __call__(self, batch):
        # speaker_id, text, mel, cwt_spec, cwt_mean, cwt_std, uv, energy, attn_prior, wav
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(-1) for x in batch]),
            dim=0, descending=True)
        
        max_text_len = max([len(x[1]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[9].size(-1) for x in batch])
        
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        
        sid = torch.LongTensor(len(batch))
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        cwt_spec_padded = torch.FloatTensor(len(batch), max_spec_len, batch[0][3].size(1))
        cwt_mean_padded = torch.FloatTensor(len(batch))
        cwt_std_padded = torch.FloatTensor(len(batch))
        uv_padded = torch.FloatTensor(len(batch), max_spec_len)
        energy_padded = torch.FloatTensor(len(batch), max_spec_len)
        attn_prior_padded = torch.FloatTensor(len(batch), max_text_len, max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), max_wav_len)
        
        sid.zero_()
        text_padded.zero_()
        spec_padded.zero_()
        cwt_spec_padded.zero_()
        cwt_mean_padded.zero_()
        cwt_std_padded.zero_()
        uv_padded.zero_()
        energy_padded.zero_()
        attn_prior_padded.zero_()
        wav_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            sid[i] = row[0]
            
            text = row[1]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            
            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            
            cwt_spec = row[3]
            cwt_spec_padded[i, :cwt_spec.size(0), :] = cwt_spec
            
            cwt_mean = row[4]
            cwt_mean_padded[i] = cwt_mean
            
            cwt_std = row[5]
            cwt_std_padded[i] = cwt_std
            
            uv = row[6]
            uv_padded[i, :uv.size(0)] = uv
            
            energy = row[7]
            energy_padded[i, :energy.size(0)] = energy
            
            attn_prior = row[8]
            attn_prior_padded[i, :text.size(0), :spec.size(1)] = attn_prior
            
            wav = row[9]
            wav_padded[i, :wav.size(0)] = wav
            
        return (
            sid, 
            text_padded, 
            text_lengths, 
            max_text_len, 
            spec_padded, 
            spec_lengths, 
            max_spec_len, 
            cwt_spec_padded, 
            cwt_mean_padded, 
            cwt_std_padded, 
            uv_padded, 
            energy_padded, 
            attn_prior_padded, 
            wav_padded
        )


    
class AudioTextDataset(Dataset):
    def __init__(self, file_path, preprocess_config):
        super(Dataset, self).__init__()
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.processor = AudioTextProcessor(preprocess_config, False)
        with open(file_path, 'r', encoding='utf8') as f:
            self.lines = f.read().split('\n')
            self.lines = [l for l in self.lines if len(l) > 0]
            tmp = []
            lengths = []
            for line in self.lines:
                audiopath, _, text = line.split('|')
                if len(text.strip()) > 1:
                    tmp.append(line)
                    lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            self.lines = tmp
            self.lengths = lengths
            
    def get_values(self, line):
        values = self.processor(line)
        values[0] = torch.LongTensor([values[0]]).long()
        values[1] = torch.LongTensor(values[1]).long()
        values[2] = torch.from_numpy(values[2]).float()
        values[3] = torch.from_numpy(values[3]).float()
        values[4] = torch.FloatTensor([values[4]]).float()
        values[5] = torch.FloatTensor([values[5]]).float()
        values[6] = torch.from_numpy(values[6]).float()
        values[7] = torch.from_numpy(values[7]).float()
        values[8] = torch.from_numpy(values[8]).float()
        values[9] = torch.from_numpy(values[9]).float()
        return values
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, index):
        return self.get_values(self.lines[index])

    
    
class AudioTextProcessor(object):
    def __init__(self, preprocess_config, preprocessing=False):
        self.preprocess_config = preprocess_config
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        self.use_intersperse = preprocess_config["preprocessing"]["text"]["use_intersperse"]
        self.preprocessing = preprocessing
        
        assert preprocess_config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        
        # self.STFT = Audio.stft.TacotronSTFT(
        #     preprocess_config["preprocessing"]["stft"]["filter_length"],
        #     preprocess_config["preprocessing"]["stft"]["hop_length"],
        #     preprocess_config["preprocessing"]["stft"]["win_length"],
        #     preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        #     preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        # )
        
        if not preprocessing:
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                self.energy_mean, self.energy_std = stats["energy"][2:]
            
    def normalize(self, values, mean, std):
        return (values - mean) / std
        
    def load_audio(self, wav_path):
        wav_raw, _ = librosa.load(wav_path, self.sampling_rate, mono=True)
        _, index = librosa.effects.trim(
            wav_raw, top_db=self.trim_top_db, 
            frame_length=self.filter_length, 
            hop_length=self.hop_length)
        wav_raw = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / self.hop_length
        return wav_raw.astype(np.float32), int(duration)
    
    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]
    
    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)
    
    def get_mel_energy(self, y):
        mel, energy = mel_spectrogram_torch(
            torch.from_numpy(y).unsqueeze(0), 
            n_fft=self.filter_length, 
            num_mels=self.n_mel_channels, 
            sampling_rate=self.sampling_rate, 
            hop_size=self.hop_length , 
            win_size=self.filter_length, 
            fmin=0, fmax=self.sampling_rate//2)
        mel, energy = mel.squeeze(0).numpy(), energy.squeeze(0).numpy()
        return mel.astype(np.float32), energy.astype(np.float32)
        
    def process_utterance(self, line):
        wav_path, speaker_id, text = line.split('|')
        text = text_to_sequence(text)
        if not self.preprocessing and self.use_intersperse:
            text = tools.intersperse(text, 0)
        
        speaker_id = int(speaker_id)
        wav, duration = self.load_audio(wav_path)
    
        # mel, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel, energy = self.get_mel_energy(wav)
        
        mel, energy = mel[:, : duration], energy[: duration]
        f0, _ = pitch_utils.get_pitch(wav, mel.T, self.preprocess_config)
        cwt_spec, _, cwt_mean_std = pitch_utils.get_f0cwt(f0[:duration])
        cwt_spec, cwt_mean, cwt_std = cwt_spec[:duration], cwt_mean_std[0], cwt_mean_std[1]
        
        energy = self.remove_outlier(energy)
        
        attn_prior = self.beta_binomial_prior_distribution(
            mel.shape[1],
            len(text),
            self.beta_binomial_scaling_factor,
        )
        
        if not self.preprocessing:
            _, uv = pitch_utils.norm_interp_f0(
                f0, self.preprocess_config["preprocessing"]["pitch"])
            energy = self.normalize(energy, self.energy_mean, self.energy_std)
            uv = uv.astype(np.float32)
        else:
            uv = None
        
        return (
            speaker_id, 
            text, 
            mel.astype(np.float32), 
            cwt_spec.astype(np.float32), 
            cwt_mean.astype(np.float32), 
            cwt_std.astype(np.float32), 
            uv, 
            energy.astype(np.float32), 
            attn_prior.astype(np.float32), 
            wav.astype(np.float32)
        )
    
    def __call__(self, line):
        return list(self.process_utterance(line))



class StatParser(AudioTextProcessor):
    def __init__(self, preprocess_config, preprocessing=True):
        super(StatParser, self).__init__(preprocess_config, preprocessing)
        self.corpus_path = preprocess_config['path']['corpus_path']
        self.row_path = preprocess_config['path']['raw_path']
        self.out_dir = preprocess_config['path']['preprocessed_path']
        self.val_size = preprocess_config['preprocessing']['val_size']
        self.energy_normalization = preprocess_config['preprocessing']['energy']['normalization']
        self.energy_scaler = StandardScaler()        
        with open(self.row_path, 'r', encoding='utf8') as f:
            self.lines = f.read().split('\n')
            self.lines = [l for l in self.lines if len(l) > 0]
            self.tmp = []
            self.energies = []
            
    def normalize(self, values, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for value in values:
            value = (value - mean) / std
            max_value = max(max_value, max(value))
            min_value = min(min_value, min(value))
        return min_value, max_value
        
    def __call__(self):
        for line in tqdm(self.lines, total=len(self.lines)):
            line = line.split('|')
            line[0] = os.path.join(self.corpus_path, line[0])
            tmp = [line[0], str(0), line[-2]]
            line = '|'.join(tmp)
            
            # try:
            values = self.process_utterance(line)
            # except:
                # print(line)
                
            if values is None:
                continue
            _, _, _, _, _, _, _, energy, _, _ = values
            if len(energy) < 1:
                continue
                
            self.energy_scaler.partial_fit(energy.reshape((-1, 1)))
            self.tmp.append(line)
            self.energies.append(energy)
            
        if self.energy_normalization:
            energy_mean = self.energy_scaler.mean_[0]
            energy_std = self.energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1
            
        energy_min, energy_max = self.normalize(
            self.energies, energy_mean, energy_std)
            
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))
    
        with open(os.path.join(self.out_dir, 'train.txt'), 'w', encoding='utf8') as f:
            trn_lines = self.tmp[:-self.val_size]
            for line in trn_lines:
                f.write(f"{line}\n")
        with open(os.path.join(self.out_dir, 'val.txt'), 'w', encoding='utf8') as f:
            val_lines = self.tmp[-self.val_size:]
            for line in val_lines:
                f.write(f"{line}\n")

                
class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
  
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
  
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
  
            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)
  
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
  
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
  
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size