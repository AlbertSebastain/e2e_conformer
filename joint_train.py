from __future__ import print_function
import argparse
import os
import math
import random
import sys
import shutil
import psutil
import time 
import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import fake_opt

from options.train_options import TrainOptions
from model.enhance_model import EnhanceModel
from model.feat_model import FFTModel, FbankModel
#from model.e2e_model import ShareE2E
from model.e2e_model import E2E
#from model.gan_model import GANModel, GANLoss, CORAL
from model.gan_model import GANModel, GANLoss
from model.e2e_common import set_requires_grad
from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed) 


def compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model):
    enhance_model.eval()
    feat_model.eval() 
    torch.set_grad_enabled(False)
    ##print(enhance_model.state_dict())
    enhance_cmvn_file = os.path.join(opt.exp_path, 'enhance_cmvn.npy')
    for i, (data) in enumerate(train_loader, start=0):
        utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes,clean_angles,mix_angles,cmvn = data
        enhance_out = enhance_model(mix_inputs, mix_log_inputs, input_sizes) 
        enhance_cmvn = feat_model.compute_cmvn(enhance_out, input_sizes)
        if enhance_cmvn is not None:
            np.save(enhance_cmvn_file, enhance_cmvn)
            print('save enhance_cmvn to {}'.format(enhance_cmvn_file))
            break
    enhance_cmvn = torch.FloatTensor(enhance_cmvn)
    enhance_model.train()
    feat_model.train()
    torch.set_grad_enabled(True)  
    return enhance_cmvn
    
         
def main():    
    opt = TrainOptions().parse()   
    if opt.exp_path == None:
        opt = fake_opt.joint_train() 
    #opt.name = sys.argv[1]
    #temp_root = '/usr/home/shi/projects/e2e_speech_project/data_model'
    #opt.exp_path = os.path.join(temp_root,opt.name)
    #opt.works_dir = opt.exp_path
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
     
    visualizer = Visualizer(opt)  
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(['train/acc', 'val/acc'], 'acc.png')

    if opt.isGAN:
        loss_report = visualizer.add_plot_report(['train/loss', 'val/loss', 'train/gan_loss','train/enhance_loss', 'val/enhance_loss', 'train/loss_D'], 'loss.png')
    else:
        loss_report = visualizer.add_plot_report(['train/loss', 'val/loss', 'train/enhance_loss', 'val/enhance_loss'], 'loss.png')
    # data
    logging.info("Building dataset.")
    train_data = opt.train_folder
    dev_data = opt.dev_folder
    #train_data = sys.argv[2]
    #dev_data = sys.argv[3]
    if 'mct' in opt.name:
        opt.MCT = True
    else:
        opt.MCT = False
    train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, train_data), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = train_data)
    val_dataset   = MixSequentialDataset(opt, os.path.join(opt.dataroot, dev_data), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = dev_data)
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
    train_loader  = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader    = MixSequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    
    # Setup an model
    lr = opt.lr
    eps = opt.eps
    iters = opt.iters   
    best_acc = opt.best_acc 
    best_loss = opt.best_loss  
    start_epoch = opt.start_epoch
    
    enhance_model_path = None
    if (opt.enhance_resume != None) & (opt.joint_resume == None):
        enhance_model_path = os.path.join(opt.works_dir, opt.enhance_resume)
        if os.path.isfile(enhance_model_path):
            enhance_model = EnhanceModel.load_model(enhance_model_path, 'enhance_state_dict', opt)
        else:
            print("no checkpoint found at {}".format(enhance_model_path))     
    
    asr_model_path = None
    if (opt.asr_resume != None) & (opt.joint_resume == None):
        asr_model_path = os.path.join(opt.works_dir, opt.asr_resume)
        if os.path.isfile(asr_model_path):
            #asr_model = ShareE2E.load_model(asr_model_path, 'asr_state_dict', opt)
            asr_model = E2E.load_model(asr_model_path,'asr_state_dict',opt)
        else:
            print("no checkpoint found at {}".format(asr_model_path))  
                                        
    joint_model_path = None
    if opt.joint_resume != None:
        joint_model_path = os.path.join(opt.works_dir, opt.joint_resume)
        if os.path.isfile(joint_model_path):
            package = torch.load(joint_model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)  
            best_acc = package.get('best_acc', 0)      
            best_loss = package.get('best_loss', float('inf'))
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0)) - 1   
            print('joint_model_path {} and iters {}'.format(joint_model_path, iters))        
            ##loss_report = package.get('loss_report', loss_report)
            ##visualizer.set_plot_report(loss_report, 'loss.png')
        else:
            print("no checkpoint found at {}".format(joint_model_path))
    if joint_model_path is not None or enhance_model_path is None:     
        enhance_model = EnhanceModel.load_model(joint_model_path, 'enhance_state_dict', opt)    
    if joint_model_path is not None or asr_model_path is None:  
        #asr_model = ShareE2E.load_model(joint_model_path, 'asr_state_dict', opt) 
        asr_model = E2E.load_model(asr_model_path,'asr_state_dict',opt)    
    feat_model = FbankModel.load_model(joint_model_path, 'fbank_state_dict', opt) 
    if opt.isGAN:
        if joint_model_path != None:
            gan_model = GANModel.load_model(joint_model_path, 'gan_state_dict', opt)
        elif opt.gan_resume != None:
            gan_path = os.path.join(opt.works_dir,opt.gan_resume)
            gan_model = GANModel.load_model(gan_path,'gan_state_dict',opt)
        else:
            gan_model = GANModel.load_model(opt.gan_resume,'gan_state_dict',opt)
    ##set_requires_grad([enhance_model], False)    
    
    # Setup an optimizer
    enhance_parameters = filter(lambda p: p.requires_grad, enhance_model.parameters())
    asr_parameters = filter(lambda p: p.requires_grad, asr_model.parameters())
    if opt.isGAN:
        gan_parameters = filter(lambda p: p.requires_grad, gan_model.parameters())   
    if opt.opt_type == 'adadelta':
        enhance_optimizer = torch.optim.Adadelta(enhance_parameters, rho=0.95, eps=eps)
        asr_optimizer = torch.optim.Adadelta(asr_parameters, rho=0.95, eps=eps)
        if opt.isGAN:
            gan_optimizer = torch.optim.Adadelta(gan_parameters, rho=0.95, eps=eps)
    elif opt.opt_type == 'adam':
        enhance_optimizer = torch.optim.Adam(enhance_parameters, lr=lr, betas=(opt.beta1, 0.999))   
        asr_optimizer = torch.optim.Adam(asr_parameters, lr=lr, betas=(opt.beta1, 0.999)) 
        if opt.isGAN:                      
            gan_optimizer = torch.optim.Adam(gan_parameters, lr=lr, betas=(opt.beta1, 0.999))
    if opt.isGAN:
        criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan).to(device)
        gan_data_file = os.path.join(opt.exp_path,'gan_data.csv')
        if(os.path.isfile(gan_data_file) & (opt.joint_resume != None)):
            gan_data = pd.read_csv(gan_data_file)
        else:
            gan_data = pd.DataFrame(columns = ['generator','discriminator'])
    if opt.MCT == True:
        fbank_path = os.path.join(opt.exp_path,'fbank_mct_cmvn.npy')
    else:
        fbank_path = os.path.join(opt.exp_path,'fbank_cmvn.npy')
    if os.path.isfile(fbank_path):

        fbank_cmvn = np.load(fbank_path)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes,clean_angels,mix_angles,cmvn = data
            fbank_cmvn = feat_model.compute_cmvn(clean_inputs, input_sizes)
            if feat_model.cmvn_processed_num >= feat_model.cmvn_num:
                #if fbank_cmvn is not None:
                fbank_cmvn = feat_model.compute_cmvn(clean_inputs, input_sizes)
                np.save(fbank_path, fbank_cmvn)
                print('save fbank_cmvn to {}'.format(fbank_path))
                break
    fbank_cmvn = torch.FloatTensor(fbank_cmvn)
    #fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
    fbank_cmvn = torch.FloatTensor(fbank_cmvn)
    # Training	
    enhance_cmvn_file = os.path.join(opt.exp_path,'enhance_cmvn.npy')
    if (os.path.isfile(enhance_cmvn_file)):
        enhance_cmvn = np.load(enhance_cmvn_file)
        enhance_cmvn = torch.FloatTensor(enhance_cmvn)
    else:
        enhance_cmvn = compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model) 
    sample_rampup = utils.ScheSampleRampup(opt.sche_samp_start_iter, opt.sche_samp_final_iter, opt.sche_samp_final_rate)  
    sche_samp_rate = sample_rampup.update(iters)
    
    enhance_model.train()
    feat_model.train()
    asr_model.train() 
    if opt.isGAN:
        gan_model.train()    
    fstlm = None          	                    
    for epoch in range(start_epoch, opt.epochs):               
        if epoch > opt.shuffle_epoch:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)  
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes,clean_angels,mix_angles,cmvn = data
            enhance_out = enhance_model(mix_inputs, mix_log_inputs, input_sizes) 
            enhance_feat = feat_model(enhance_out)
            clean_feat = feat_model(clean_inputs)
            mix_feat = feat_model(mix_inputs)
            if opt.enhance_loss_type == 'L2':
                enhance_loss = F.mse_loss(enhance_feat, clean_feat.detach())
            elif opt.enhance_loss_type == 'L1':
                enhance_loss = F.l1_loss(enhance_feat, clean_feat.detach())
            elif opt.enhance_loss_type == 'smooth_L1':
                enhance_loss = F.smooth_l1_loss(enhance_feat, clean_feat.detach())
            enhance_loss = opt.enhance_loss_lambda * enhance_loss
                
            #loss_ctc, loss_att, acc, clean_context, mix_context = asr_model(clean_feat, enhance_feat, targets, input_sizes, target_sizes, sche_samp_rate, enhance_cmvn) 
            clean_feature = feat_model(clean_inputs,fbank_cmvn)
            enhance_feature = feat_model(enhance_out,enhance_cmvn)
            loss_ctc, loss_att, acc = asr_model(enhance_feature, targets, input_sizes, target_sizes, sche_samp_rate)
            #nbest_hyps = asr_model.recognize(enhance_feature, opt, opt.char_list, rnnlm=rnnlm, fstlm=fstlm)
            #mix_context = nbest_hyps[0]['yseq'][1:]
            #nbest_hyps = asr_model.recognize(clean_feature, opt, opt.char_list, rnnlm=rnnlm, fstlm=fstlm)
            #clean_context = nbest_hyps[0]['yseq'][1:]
            #coral_loss = opt.coral_loss_lambda * CORAL(clean_context, mix_context)              
            asr_loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
            coral_loss = 0
            loss = asr_loss + enhance_loss + coral_loss
                    
            if opt.isGAN:
                set_requires_grad([gan_model], False)
                gan_model.eval()
                if opt.netD_type == 'pixel':
                    fake_AB = torch.cat((mix_feat, enhance_feat), 2)
                else:
                    #fake_AB = enhance_feat
                    fake_AB = enhance_feature
                #fake_AB_feat = feat_model(fake_AB,enhance_cmvn)
                #gan_loss = opt.gan_loss_lambda * criterionGAN(gan_model(fake_AB,enhance_cmvn), True)
                gan_loss_G = criterionGAN(gan_model(fake_AB), True)
                gan_loss = opt.gan_loss_lambda * gan_loss_G
                loss += gan_loss
                                              
            enhance_optimizer.zero_grad()
            asr_optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()          
            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                enhance_optimizer.step()
                asr_optimizer.step()                
            
            if opt.isGAN:
                set_requires_grad([gan_model], True)   
                gan_model.train()
                gan_optimizer.zero_grad()
                if opt.netD_type == 'pixel':
                    fake_AB = torch.cat((mix_feat, enhance_feat), 2)
                    real_AB = torch.cat((mix_feat, clean_feat), 2)
                else:
                    #fake_AB = enhance_feat
                    #real_AB = clean_feat
                    fake_AB = enhance_feature
                    real_AB = clean_feature
                #loss_D_real = criterionGAN(gan_model(real_AB.detach(), enhance_cmvn), True)
                #loss_D_fake = criterionGAN(gan_model(fake_AB.detach(), enhance_cmvn), False)
                loss_D_real = criterionGAN(gan_model(real_AB.detach()), True)
                loss_D_fake = criterionGAN(gan_model(fake_AB.detach()), False)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(gan_model.parameters(), opt.grad_clip)
                if math.isnan(grad_norm):
                    logging.warning('grad norm is nan. Do not update model.')
                else:
                    gan_optimizer.step()
                  
            iters += 1
            #errors = {'train/loss': loss.item(), 'train/loss_ctc': loss_ctc.item(), 
                      #'train/acc': acc, 'train/loss_att': loss_att.item(), 
                      #'train/enhance_loss': enhance_loss.item(), 'train/coral_loss': coral_loss.item()}
            errors = {'train/loss': loss.item(), 'train/loss_ctc': loss_ctc.item(), 
                      'train/acc': acc, 'train/loss_att': loss_att.item(), 
                      'train/enhance_loss': enhance_loss.item()}
            if opt.isGAN:
                errors['train/loss_D'] = loss_D.item()
                errors['train/gan_loss'] = opt.gan_loss_lambda * gan_loss.item()  
              
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'fbank_state_dict': feat_model.state_dict(), 
                         'enhance_state_dict': enhance_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                if opt.isGAN:
                    state['gan_state_dict'] = gan_model.state_dict()
                    gan_data = gan_data.append({'generator':gan_loss_G,'discriminator':loss_D},ignore_index = True)
                filename='latest'
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                # for name, para in enhance_model.named_parameters():
                #     logging.info('name.'+str(name))
                #     logging.info('para.')
                #     logging.info(para[0:3,0:3])
                #     logging.info('paramsgrad.')
                #     logging.info(para.grad[0:3,0:3]) 
                #     break  
            if iters % opt.validate_freq == 0:
                sche_samp_rate = sample_rampup.update(iters)
                print("iters {} sche_samp_rate {}".format(iters, sche_samp_rate))    
                enhance_model.eval() 
                feat_model.eval() 
                asr_model.eval()
                if opt.isGAN:
                    gan_model.eval()
                torch.set_grad_enabled(False)                
                num_saved_attention = 0 
                for i, (data) in tqdm(enumerate(val_loader, start=0)):
                    utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes,clean_angles,mix_angles,cmvn = data
                    enhance_out = enhance_model(mix_inputs, mix_log_inputs, input_sizes)                         
                    enhance_feat = feat_model(enhance_out)
                    clean_feat = feat_model(clean_inputs)
                    mix_feat = feat_model(mix_inputs)
                    val_enhance_feature = feat_model(enhance_out,enhance_cmvn)
                    val_clean_feature = feat_model(clean_inputs,fbank_cmvn)
                    if opt.enhance_loss_type == 'L2':
                        enhance_loss = F.mse_loss(enhance_feat, clean_feat.detach())
                    elif opt.enhance_loss_type == 'L1':
                        enhance_loss = F.l1_loss(enhance_feat, clean_feat.detach())
                    elif opt.enhance_loss_type == 'smooth_L1':
                        enhance_loss = F.smooth_l1_loss(enhance_feat, clean_feat.detach())
                    if opt.isGAN:
                        set_requires_grad([gan_model], False)
                        if opt.netD_type == 'pixel':
                            fake_AB = torch.cat((mix_feat, enhance_feat), 2)
                        else:
                            #fake_AB = enhance_feat
                            fake_AB = val_enhance_feature
                        #gan_loss = criterionGAN(gan_model(fake_AB, enhance_cmvn), True)
                        gan_loss = criterionGAN(gan_model(fake_AB), True)
                        enhance_loss += opt.gan_loss_lambda * gan_loss
                        
                    #loss_ctc, loss_att, acc, clean_context, mix_context = asr_model(clean_feat, enhance_feat, targets, input_sizes, target_sizes, 0.0, enhance_cmvn)
                    #val_enhance_feature = feat_model(enhance_out,enhance_cmvn)
                    loss_ctc,loss_att,acc = asr_model(val_enhance_feature, targets, input_sizes, target_sizes, sche_samp_rate)                     
                    asr_loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
                    enhance_loss = opt.enhance_loss_lambda * enhance_loss
                    loss = asr_loss + enhance_loss                          
                    errors = {'val/loss': loss.item(), 'val/loss_ctc': loss_ctc.item(), 
                              'val/acc': acc, 'val/loss_att': loss_att.item(),
                              'val/enhance_loss': enhance_loss.item()}
                    if opt.isGAN:        
                        errors['val/gan_loss'] = opt.gan_loss_lambda * gan_loss.item()  
                    visualizer.set_current_errors(errors)
                
                    if opt.num_save_attention > 0 and opt.mtlalpha != 1.0:
                        if num_saved_attention < opt.num_save_attention:
                            #att_ws = asr_model.calculate_all_attentions(enhance_feat, targets, input_sizes, target_sizes, enhance_cmvn) 
                            att_ws = asr_model.calculate_all_attentions(val_enhance_feature, targets, input_sizes, target_sizes)                             
                            for x in range(len(utt_ids)):
                                att_w = att_ws[x]
                                utt_id = utt_ids[x]
                                file_name = "{}_ep{}_it{}.png".format(utt_id, epoch, iters)
                                dec_len = int(target_sizes[x])
                                enc_len = int(input_sizes[x]) 
                                visualizer.plot_attention(att_w, dec_len, enc_len, file_name) 
                                num_saved_attention += 1
                                if num_saved_attention >= opt.num_save_attention:   
                                    break 
                enhance_model.train()
                feat_model.train()
                asr_model.train() 
                if opt.isGAN:
                    gan_model.train()
                torch.set_grad_enabled(True)  
				
                visualizer.print_epoch_errors(epoch, iters)  
                acc_report = visualizer.plot_epoch_errors(epoch, iters, 'acc.png') 
                loss_report = visualizer.plot_epoch_errors(epoch, iters, 'loss.png') 
                val_loss = visualizer.get_current_errors('val/loss')
                val_acc = visualizer.get_current_errors('val/acc') 
                filename = None                
                if opt.criterion == 'acc' and opt.mtl_mode is not 'ctc':
                    if val_acc < best_acc:
                        logging.info('val_acc {} > best_acc {}'.format(val_acc, best_acc))
                        opt.eps = utils.adadelta_eps_decay(asr_optimizer, opt.eps_decay)
                    else:
                        filename='model.acc.best'                    
                    best_acc = max(best_acc, val_acc)
                    logging.info('best_acc {}'.format(best_acc))  
                elif opt.criterion == 'loss':
                    if val_loss > best_loss:
                        logging.info('val_loss {} > best_loss {}'.format(val_loss, best_loss))
                        opt.eps = utils.adadelta_eps_decay(asr_optimizer, opt.eps_decay)
                    else:
                        filename='model.loss.best'    
                    best_loss = min(val_loss, best_loss)
                    logging.info('best_loss {}'.format(best_loss))                  
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'fbank_state_dict': feat_model.state_dict(), 
                         'enhance_state_dict': enhance_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                if opt.isGAN:
                    state['gan_state_dict'] = gan_model.state_dict()
                utils.save_checkpoint(state, opt.exp_path, filename=filename)                  
                visualizer.reset()  
                enhance_cmvn = compute_cmvn_epoch(opt, train_loader, enhance_model, feat_model)  
    if opt.isGAN:
        gan_data.to_csv(gan_data_file,index = False)           
if __name__ == '__main__':
    main()
