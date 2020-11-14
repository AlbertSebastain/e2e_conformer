import os
import sys

from numpy.core.fromnumeric import argmax
import kaldi_io
import librosa
import numpy as np
import scipy.signal
import scipy.io

import torchaudio
import math
import random
from random import choice
from multiprocessing import Pool
import scipy.io.wavfile as wav
from os import walk

def load_audio(path):
    sound, _ = torchaudio.load_wav(path)
    sound = sound.numpy()
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    sound = sound / (2**15)
    return sound

def load_audio_noise(path):
    name = path.split(os.sep)[-1].split(".")[0]
    sound = scipy.io.loadmat(path)[name]
    sound = sound.squeeze()
    sound = sound / (2**15)
    sound = sound.astype('float32')
    return sound

def MakeMixture(speech, noise, db):
    if speech is None or noise is None:
        return None
    if np.sum(np.square(noise)) < 1.0e-6:
        return None

    spelen = speech.shape[0]

    exnoise = noise
    while exnoise.shape[0] < spelen:
        exnoise = np.concatenate([exnoise, noise], 0)
    noise = exnoise
    noilen = noise.shape[0]

    elen = noilen - spelen - 1
    if elen > 1:
        s = round(random.randint(0, elen - 1))
    else:
        s = 0
    e = s + spelen

    noise = noise[s:e]

    try:
        noi_pow = np.sum(np.square(noise))
        if noi_pow > 0:
            noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
        else:
            print(noi_pow, np.square(noise), "error")
            return None
    except:
        return None

    nnoise = noise * noi_scale
    mixture = speech + nnoise
    mixture = mixture.astype("float32")
    return mixture


def make_feature(wav_path_list, noise_wav_list, feat_dir, thread_num, argument=False, repeat_num=1,data_type = ''):
    mag_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats{1}.ark,{0}/feats{1}.scp".format(feat_dir, thread_num)
    ang_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/angles{1}.ark,{0}/angles{1}.scp".format(feat_dir, thread_num)
    if argument:
        fwrite = open(os.path.join(feat_dir, "db" + str(thread_num)), "a")
    f_mag = kaldi_io.open_or_fd(mag_ark_scp_output, "wb")
    f_ang = kaldi_io.open_or_fd(ang_ark_scp_output, "wb")

    for num in range(repeat_num):
        for tmp in wav_path_list:
            uttid, wav_path = tmp
            clean = load_audio(wav_path)
            y = None
            while y is None:
                if argument:
                    noise_path = choice(noise_wav_list)
                    n = load_audio_noise(noise_path)
                    #n = load_audio_noise(noise_wav_path)
                    noise_clean = np.random.randint(10,size = 1)
                    if (noise_clean == 0) & ('test' not in data_type):
                        y = clean
                        db = np.inf

                        noise_name = 'none'
                    else:
                        db = np.random.uniform(low=0, high=20)
                        y = MakeMixture(clean, n, db)
                        noise_name = noise_path.split('/')[-1]
                    uttid_new = uttid + "__mix{}".format(num)

                    #print(uttid_new + " " + str(db) + "\n")
                    fwrite.write(uttid_new + " " + str(db)+" "+noise_name + "\n")
                else:
                    y = clean
                    uttid_new = uttid
            # STFT
            if y is not None:
                D = librosa.stft(y, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
                spect = np.abs(D)
                angle = np.angle(D)
                ##feat = np.concatenate((spect, angle), axis=1)
                ##feat = feat.transpose((1, 0))
                kaldi_io.write_mat(f_mag, spect.transpose((1, 0)), key=uttid_new)
                kaldi_io.write_mat(f_ang, angle.transpose((1, 0)), key=uttid_new)
            else:
                print(noise_path, tmp, "error")

    f_mag.close()
    f_ang.close()
    if argument:
        fwrite.close()

def prepare_spec(data_dir,feat_dir,noise_repeat_num,data_type,noise_path):
    feat_dir = os.path.join(feat_dir, data_type)
    clean_feat_dir = os.path.join(feat_dir, "clean")
    mix_feat_dir = os.path.join(feat_dir, data_type)
    mix_feat_dir = os.path.join(feat_dir, "mix")
    if not os.path.exists(mix_feat_dir):
        os.makedirs(mix_feat_dir)

        # 读取clean_wav的路径
    clean_wav_list = []
    data_dir = os.path.join(data_dir, data_type)
    clean_wav_scp = os.path.join(data_dir, "wav.scp")  # 在data_dir 目录下，有一个clean_wav.scp 文件
    with open(clean_wav_scp, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip().replace("\n", "")
            uttid, wav_path = line.split()
            clean_wav_list.append((uttid, wav_path))
    print(">> clean_wav_list len:", len(clean_wav_list))
    noise_wav_list = []
    if 'test' in data_type:
        noise_name = ['pink.mat','factory2.mat','buccaneer2.mat','destroyerops.mat','m109.mat']

    else:
        noise_name = ['white.mat','factory1.mat','buccaneer1.mat','f16.mat','destroyerengine.mat','leopard.mat','machinegun.mat','volvo.mat','hfchannel.mat']
    for noises in noise_name:
        noise_wav_list.append(os.path.join(noise_path,noises))
    print(">> noise_wav_list len",len(noise_wav_list))
    print(">>noise length", len(noise_wav_list))

    # 使用八个线程
    threads_num = 8

    wav_num = len(clean_wav_list)
    print(">> Parent process %s." % os.getpid())
    p = Pool()
    for i in range(threads_num):
        wav_path_tmp_list = clean_wav_list[int(i * wav_num / threads_num) : int((i + 1) * wav_num / threads_num)]
        p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num,data_type))
        #make_feature(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num)
    print(">> Waiting for all subprocesses done...")
    p.close()
    p.join()
    print(">> All subprocesses done.")

    command_line = "cat {}/feats*.scp > {}/mix_feats.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)
    command_line = "cat {}/angles*.scp > {}/mix_angles.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)
    command_line = "cat {}/db* > {}/db.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)

def main():

    # 输入参数
    # data_dir = sys.argv[1]  # data根目录 e.g. data
    # feat_dir = sys.argv[2]  # ark文件，输出路径 e.g. data/feats
    # noise_repeat_num = int(sys.argv[3])  # noise重复次数
    # data_type = sys.argv[4]  # 数据类型 e.g. train / test / dev
    # db = sys.argv[5]
    # noise_path = sys.argv[6]
    # print(data_dir)
    # print(feat_dir)
    # print(noise_repeat_num)
    # print(data_type)
    # print(db)
    # print(noise_path)
    #dbs = [-5,5,10]
    #dbdir=['datadb_n5','datadb_p5','datadb_p10']
    #noisedir=['data_w','data_f','data_d']
    #noise_type=['white.mat','factory1.mat','destroyerengine.mat']
    data_root = '/usr/home/shi/projects/data_aishell/data'
    data_types = ['mix_test_unmatch','mix_train_clean_match','mix_dev_clean_match']
    data_types = ['mix_train_clean_match','mix_dev_clean_match']
    #data_types = ['mix_test_match']
    #data_types = ['small_train','small_test']
    noise_root = '/usr/home/shi/projects/data_aishell/data/noise/'
    noise_repeat_num = 1
    for data_type in data_types:
                #data_dir = os.path.join(data_root,dbdir[i],noisedir[j])
        feat_dir = data_root
                #noise_path = os.path.join(noise_root,noise_type[j])
        prepare_spec(data_root,feat_dir,noise_repeat_num,data_type,noise_root)

if __name__ == "__main__":
    main()
