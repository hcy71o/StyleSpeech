from multiprocessing.sharedctypes import Value
import audio as Audio
from text import _clean_text
import numpy as np
import librosa
import os
from pathlib import Path
from scipy.io.wavfile import write
from joblib import Parallel, delayed
import tgt
import pyworld as pw
from preprocessors.utils import remove_outlier, get_alignment, average_by_duration
from scipy.interpolate import interp1d
import json
from models.Wav2vec2 import Wav2vec2
import torch


def write_single(output_folder, wav_fname, text, resample_rate, top_db=None):
    data, sample_rate = librosa.load(wav_fname, sr=None)
    # trim audio
    wav_fname = wav_fname.split('/')[-1]
    
    if top_db is not None:
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    else:
        trimmed = data
    # resample audio
    resampled = librosa.resample(trimmed, sample_rate, resample_rate)
    y = (resampled * 32767.0).astype(np.int16)
    
    #* for wav2vec2.0
    resampled_16k = librosa.resample(trimmed, sample_rate, 16000)
    y_16k = (resampled_16k * 32767.0).astype(np.int16)
    dir_sr = output_folder.split('/')[-2]
    output_folder_16k = os.path.join(output_folder.split(dir_sr)[0] + 'wav16' + output_folder.split(dir_sr)[1])
    if not os.path.exists(output_folder_16k):
        os.makedirs(output_folder_16k, exist_ok=True)
    target_wav16_fname = os.path.join(output_folder_16k, wav_fname)
    write(target_wav16_fname, 16000, y_16k)

    
    target_wav_fname = os.path.join(output_folder, wav_fname)
    target_txt_fname = os.path.join(output_folder, wav_fname.replace('.wav', '.txt'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    write(target_wav_fname, resample_rate, y)
    with open(target_txt_fname, 'wt') as f:
        f.write(text)
        f.close()
    return y.shape[0] / float(resample_rate)


def prepare_align_and_resample(data_dir, out_dir, sr):
    wav_foder_names = ['train-clean-100', 'train-clean-360']
    wavs = []
    for wav_folder in wav_foder_names:
        wav_folder = os.path.join(data_dir, wav_folder)
        wav_fname_list = [str(f) for f in list(Path(wav_folder).rglob('*.wav'))]
        output_wavs_folder_name = 'wav{}'.format(sr//1000)
        output_wavs_folder = os.path.join(out_dir, output_wavs_folder_name)
        if not os.path.exists(output_wavs_folder):
            os.mkdir(output_wavs_folder)

        for wav_fname in wav_fname_list:

            _sid = wav_fname.split('/')[-3]
            output_folder = os.path.join(output_wavs_folder, _sid)
            txt_fname = wav_fname.replace('.wav','.normalized.txt')
            with open(txt_fname, 'r') as f:
                text = f.readline().strip()
            text = _clean_text(text, ['english_cleaners'])
            wavs.append((output_folder, wav_fname, text))

    lengths = Parallel(n_jobs=10, verbose=1)(
        delayed(write_single)(wav[0], wav[1], wav[2], sr) for wav in wavs
    )


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config["sampling_rate"]

        self.n_mel_channels = config["n_mel_channels"]
        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.win_length = config["win_length"]
        self.max_wav_value = config["max_wav_value"]
        self.mel_fmin = config["mel_fmin"]
        self.mel_fmax= config["mel_fmax"]

        self.max_seq_len = config["max_seq_len"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["filter_length"],
            config["hop_length"],
            config["win_length"],
            config["n_mel_channels"],
            config["sampling_rate"],
            config["mel_fmin"],
            config["mel_fmax"],
        )
        
        self.wav2vec2 = Wav2vec2().cuda().eval()

    def write_metadata(self, data_dir, out_dir):
        metadata = os.path.join(out_dir, 'metadata.csv')
        if not os.path.exists(metadata):
            wav_fname_list = [str(f) for f in list(Path(data_dir).rglob('*.wav'))]
            lines = []
            for wav_fname in wav_fname_list:
                basename = wav_fname.split('/')[-1].replace('.wav', '')
                sid = wav_fname.split('/')[-2]
                assert sid in basename
                txt_fname = wav_fname.replace('.wav', '.txt')
                with open(txt_fname, 'r') as f:
                    text = f.readline().strip()
                    f.close()
                lines.append('{}|{}|{}'.format(basename, text, sid))
            with open(metadata, 'wt') as f:
                f.writelines('\n'.join(lines))
                f.close()

    def build_from_path(self, data_dir, out_dir):
        datas = list()
        f0 = list()
        energy = list()
        n_frames = 0
        with open(os.path.join(out_dir, 'metadata.csv'), encoding='utf-8') as f:
            basenames = []
            for line in f:
                parts = line.strip().split('|')
                basename = parts[0]
                basenames.append(basename)
        
        #* Total utt (len(basenames)): 149736
        results = Parallel(n_jobs=2, verbose=1)(
                delayed(self.process_utterance)(data_dir, out_dir, basename) for basename in basenames
            )
        results = [ r for r in results if r is not None ]
        
        for r in results:
            datas.append(r[0])
            f0.extend(r[1])
            energy.extend(r[2])
            n_frames += r[3]

        f0 = remove_outlier(f0)
        energy = remove_outlier(energy)

        f0_max = np.max(f0)
        f0_min = np.min(f0)
        f0_mean = np.mean(f0)
        f0_std = np.std(f0)
        energy_max = np.max(energy)
        energy_min = np.min(energy)
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)

        total_time = n_frames*self.hop_length/self.sampling_rate/3600
        f_json = {
            "total_time": str(total_time),
            "n_frames": str(n_frames),
            "f0_stat": [str(f0_max), str(f0_min), str(f0_mean), str(f0_std)],
            "energy_stat": [str(energy_max), str(energy_min), str(energy_mean), str(energy_std)]
        }

        with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
            json.dump(f_json, f)
        
        return datas


    def process_utterance(self, in_dir, out_dir, basename, dataset='libritts'):

        sid = basename.split('_')[0]
        wav_path = os.path.join(in_dir, 'wav{}'.format(self.sampling_rate//1000), sid, '{}.wav'.format(basename))
        tg_path = os.path.join(out_dir, 'TextGrid', sid, '{}.TextGrid'.format(basename)) 

        ''' This should be removed.. mistake
        #*Wav2vec2.0
        wav16_path = os.path.join(in_dir, 'wav16', sid, '{}.wav'.format(basename))
        wav16, sr16 = librosa.load(wav16_path, sr=None) #* preserve native sampling rate of the file
        wav16 = torch.FloatTensor(wav16).unsqueeze(0).cuda()
        
        with torch.no_grad():
            #* (1, seq_len, 756)
            latent_output = self.wav2vec2(wav16)
            #* (seq_len, 756)
            latent_output = latent_output.squeeze(0).detach().cpu().numpy()
        '''
        
        if not os.path.exists(wav_path) or not os.path.exists(tg_path):
            return None
        
        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = get_alignment(textgrid.get_tier_by_name('phones'), self.sampling_rate, self.hop_length)
        text = '{'+ '}{'.join(phone) + '}' # '{A}{B}{$}{C}', $ represents silent phones
        text = text.replace('{$}', ' ')    # '{A}{B} {C}'
        text = text.replace('}{', ' ')     # '{A B} {C}'

        if start >= end:
            return None

        # Read and trim wav files
        wav, sr = librosa.load(wav_path, sr=None) #* preserve native sampling rate of the file
        wav = wav[int(self.sampling_rate*start):int(self.sampling_rate*end)].astype(np.float32)
        
        #TODO Save latent vector with the trimmed waveforms
        wav_16 = librosa.resample(wav, sr, 16000)
        wav_16 = torch.FloatTensor(wav_16).unsqueeze(0).cuda()

        with torch.no_grad():
            #* (1, seq_len, 756)
            latent_output = self.wav2vec2(wav_16)
            #* (seq_len, 756)
            latent_output = latent_output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        # Compute fundamental frequency
        _f0, t = pw.dio(wav.astype(np.float64), self.sampling_rate, frame_period=self.hop_length/self.sampling_rate*1000)
        f0 = pw.stonemask(wav.astype(np.float64), _f0, t, self.sampling_rate)
        f0 = f0[:sum(duration)]
        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, :sum(duration)]
        energy = energy[:sum(duration)]
        
        if mel_spectrogram.shape[1] >= self.max_seq_len:
            return None
        
        # Pitch perform linear interpolation
        nonzero_ids = np.where(f0 != 0)[0]
        if len(nonzero_ids)>=2:
            interp_fn = interp1d(
                nonzero_ids,
                f0[nonzero_ids],
                fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]),
                bounds_error=False,
            )
            f0 = interp_fn(np.arange(0, len(f0)))
        # Pitch phoneme-level average
        f0 = average_by_duration(np.array(f0), np.array(duration))
        # Energy phoneme-level average
        energy = average_by_duration(np.array(energy), np.array(duration))
        if len([f for f in f0 if f != 0]) ==0 or len([e for e in energy if e != 0])==0:
            return None
        
        # Save alignment
        ali_filename = '{}-ali-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'alignment', ali_filename), duration, allow_pickle=False)
        # Save fundamental frequency
        f0_filename = '{}-f0-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)
        # Save energy
        energy_filename = '{}-energy-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)
        # Save spectrogram
        mel_filename = '{}-mel-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
        
        #* Save latent extracted from Wav2vec2.0
        latent_filename = '{}-latent-{}.npy'.format(dataset, basename)
        np.save(os.path.join(out_dir, 'latent', latent_filename), latent_output, allow_pickle=False)

        return None

        return '|'.join([basename, text, sid]), list(f0), list(energy), mel_spectrogram.shape[1]