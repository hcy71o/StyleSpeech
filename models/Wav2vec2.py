import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import librosa

'''
preprocess 단계에서 사전에 추출
이후, 추출된 emb -> pooling -> 2XFC -> latent vector
masking은 이후 pooling layer 부분에서만 고려 (FC에선 고려할 필요 X)
'''

class Wav2vec2(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        
    def forward(self, x):
        with torch.no_grad():
            features, _ = self.wav2vec2.extract_features(x)
            features = features[-1]
        return features
    

if __name__ == "__main__":
    wav2vec = Wav2vec2()
    wav, sr = librosa.load('/home/hcy71/VCTK/VCTK-Corpus/wav48/p228/p228_001.wav')
    wav = librosa.resample(wav, sr, 16000)
    wav = torch.FloatTensor(wav).unsqueeze(0)
    output = wav2vec(wav)
    print(output.shape)