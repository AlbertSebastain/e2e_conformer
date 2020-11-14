import torchaudio
import numpy as np
from torchaudio.backend.sox_backend import load_wav
def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=False)
    sound = sound.numpy()
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    #sound = sound / 65536.
    return sound
def load_wav_audio(path):
    sound,_ = load_wav(path,normalization = False)
    sound = sound.numpy()
    return sound
path = '/usr/home/shi/projects/data_aishell/dataaishell/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav'
sound = load_audio(path)
print('from load  ',sound)
sound1 = load_wav_audio(path)
