import torchaudio
import numpy as np
def load_audio(path):
    sound, rate = torchaudio.load(path, normalization=True)
    sound = sound.numpy()
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    #sound = sound / 65536.
    #print(rate)
    return sound
path = '/usr/home/shi/projects/data_aishell/dataaishell/data_aishell/wav/train/S0002/BAC009S0002W0122.wav'
souund = load_audio(path)
xxs = 1+1j
x_abs = np.abs(xxs)
x_ang = np.angle(xxs)
xx = 1j*x_abs*np.sin(x_ang)+x_abs*np.cos(x_ang)
print(x_abs,xx)
