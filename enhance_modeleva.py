from __future__ import print_function
import argparse
import os
import math
import random
import shutil
import psutil
import time 
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import fake_opt
import scipy.io.wavfile
import librosa
import os

from options.train_options import TrainOptions
from model.enhance_model import EnhanceModel
from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils
from rewav import rewav
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
opt = fake_opt.Enhance_base_train()  
opt.enhance_model = 'model.loss.best'
train_set = 'small_train'
path_enhance = '/usr/home/shi/projects/data_aishell/data/wavfile/enh'
path_init = '/usr/home/shi/projects/data_aishell/data/wavfile/initial'
train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, train_set), os.path.join(opt.dict_dir, 'train_units.txt'),train_set) 
#val_dataset   = MixSequentialDataset(opt, os.path.join(opt.dataroot, val_set), os.path.join(opt.dict_dir, 'train_units.txt'),val_set)
train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
train_loader = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
#val_loader = MixSequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
if opt.enhance_resume:
        model_path = os.path.join(opt.works_dir, opt.enhance_resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))            
            #loss_report = package.get('loss_report', loss_report)
            #visualizer.set_plot_report(loss_report, 'loss.png')
            #print('package found at {} and start_epoch {} iters {}'.format(model_path, start_epoch, iters))
enhance_model = EnhanceModel.load_model(model_path, 'enhance_state_dict', opt)
enhance_model.eval()
for i, (data) in enumerate(train_loader, start=(iters % len(train_dataset))):
    utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes, clean_angles, mix_angles, cmvn = data
    loss, enhance_out = enhance_model( mix_inputs, mix_log_inputs, input_sizes,  clean_inputs,cos_angles)
    wav_num = enhance_out.shape[0]
    for j in range(0,wav_num):
        enhance_spec = enhance_out[j].data
        #lean_angle = clean_angles[j].data
        mix_spec = mix_inputs[j].data
        mix_angle = mix_angles[j].data
        #input_enhance = (input_enhance-cmvn[0,:])/cmvn[1,:]
        uttid = utt_ids[j]
        input_size = input_sizes[j]
        #input_init = (input_init-cmvn[0,:])/cmvn[1,:]
        rewav(path_enhance,uttid,enhance_spec,mix_angle,type_wav = 'enhance',input_size = input_size)
        rewav(path_init,uttid,mix_spec,mix_angle, type_wav = 'initial',input_size = input_size)
