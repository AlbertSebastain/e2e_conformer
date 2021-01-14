#from data.labelparse import Labelparse

import math
import random


import itertools
import numpy as np
import torch
import torch.optim as optim
import os
import torch.nn.functional as F
from data.data_loader import SequentialDataset, SequentialDataLoader,BucketingSampler

from utils.visualizer import Visualizer
from utils.utils import ScheSampleRampup, save_checkpoint, adadelta_eps_decay
from tqdm import tqdm

from transformer.optimizer import NoamOpt
from transformer.nets_utils import pad_list
from e2e_asr_conformer import E2E
from conformer_options.train_conformer_options import Train_conformer_Options
import fake_opt
from model.feat_model import FbankModel
import datetime

SEED = random.randint(1, 10000)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def train():
    opt = Train_conformer_Options().parse()
    if opt.name == '':
        opt = fake_opt.asr_conf()

    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")

    visualizer = Visualizer(opt)
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(["train/acc", "val/acc"], "acc.png")
    loss_report = visualizer.add_plot_report(["train/loss", "val/loss"], "loss.png")

    train_fold = opt.train_folder
    dev_fold = opt.dev_folder
    train_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, train_fold), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = 'train',mct = opt.MCT) 
    val_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, dev_fold), os.path.join(opt.dict_dir, 'train_units.txt'),type_data = 'dev',mct = opt.MCT)    
    train_sampler = BucketingSampler(train_dataset,batch_size = opt.batch_size)
    train_loader = SequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = SequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    # add new parameters
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)

    logging.info("#input dims : " + str(opt.idim))
    logging.info("#output dims: " + str(opt.odim))
    logging.info("Dataset ready!")

    #asr_model = E2E(opt.idim, opt.odim, opt)
    asr_model = E2E(opt)
    fbank_model = FbankModel(opt)
    lr = opt.lr  # default=0.005
    eps = opt.eps  # default=1e-8
    iters = opt.iters  # default=0
    start_epoch = opt.start_epoch  # default=0
    best_loss = opt.best_loss  # default=float('inf')
    best_acc = opt.best_acc  # default=0
    model_path = None
    # convert to cuda
    #fbank_model.cuda()
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            best_acc = package.get('best_acc', 0)
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))
            iters = iters-1
            
            acc_report = package.get('acc_report', acc_report)
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(acc_report, 'acc.png')
            visualizer.set_plot_report(loss_report, 'loss.png')
            

            logging.info('Loading model {} and iters {}'.format(model_path, iters))
        else:
            print("no checkpoint found at {}".format(model_path))     
    asr_model = E2E.load_model(model_path, 'asr_state_dict',opt) 
    fbank_model = FbankModel.load_model(model_path, 'fbank_state_dict',opt) 
    parameters = filter(lambda p: p.requires_grad, itertools.chain(asr_model.parameters()))
    optimizer = torch.optim.Adam(parameters,lr = lr,betas = (opt.beta1,0.98), eps=eps)
    if opt.opt_type == 'noam':
        optimizer = NoamOpt(asr_model.adim, opt.transformer_lr, opt.transformer_warmup_steps, optimizer,iters)
    asr_model.cuda()
    print(asr_model)
    asr_model.train()
    #sample_rampup = ScheSampleRampup(opt.sche_samp_start_iter, opt.sche_samp_final_iter, opt.sche_samp_final_rate)
    #sche_samp_rate = sample_rampup.update(iters)
    #print(opt.MCT)
    if opt.MCT == True:
        fbank_cmvn_file = os.path.join(opt.exp_path,'fbank_mct_cmvn.npy')
    else:
        fbank_cmvn_file = os.path.join(opt.exp_path, 'fbank_cmvn.npy')
    if os.path.exists(fbank_cmvn_file):
            fbank_cmvn = np.load(fbank_cmvn_file)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
            if fbank_model.cmvn_processed_num >= fbank_model.cmvn_num:
                #if fbank_cmvn is not None:
                fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
                np.save(fbank_cmvn_file, fbank_cmvn)
                print('save fbank_cmvn to {}'.format(fbank_cmvn_file))
                break
    fbank_cmvn = torch.FloatTensor(fbank_cmvn)
    for epoch in range(start_epoch, opt.epochs):
        if epoch > opt.shuffle_epoch:
            print(">> Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)
        for i, (data) in enumerate(train_loader, start=(iters * opt.batch_size) % len(train_dataset)):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_features = fbank_model(inputs, fbank_cmvn)

            loss, acc = asr_model(fbank_features, input_sizes, targets,target_sizes)
            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()  # compute backwards
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning(">> grad norm is nan. Do not update model.")
            else:
                optimizer.step()

            iters += 1
            errors = {
                "train/loss": loss.item(),
                "train/acc": acc,
            }
            visualizer.set_current_errors(errors)

            # print
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {
                    "asr_state_dict": asr_model.state_dict(),
                    "opt": opt,
                    "epoch": epoch,
                    "iters": iters,
                    "eps": opt.eps,
                    "lr": opt.lr,
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "acc_report": acc_report,
                    "loss_report": loss_report,
                }
                filename = "latest"
                save_checkpoint(state, opt.exp_path, filename=filename)
            # evalutation
            if iters % opt.validate_freq == 0:
                asr_model.eval()
                torch.set_grad_enabled(False)
                pbar = tqdm(total=len(val_dataset))
                for i, (data) in enumerate(val_loader, start=0):
                    utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data

                    fbank_features = fbank_model(inputs, fbank_cmvn)
                    loss,acc = asr_model(fbank_features, input_sizes, targets,target_sizes)
                    #loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
                    errors = {
                        "val/loss": loss.item(),
                        "val/acc": acc,
                    }
                    visualizer.set_current_errors(errors)
                    pbar.update(opt.batch_size)
                pbar.close()
                asr_model.train()
                torch.set_grad_enabled(True)

                visualizer.print_epoch_errors(epoch, iters)
                acc_report = visualizer.plot_epoch_errors(epoch, iters, "acc.png")
                loss_report = visualizer.plot_epoch_errors(epoch, iters, "loss.png")
                val_loss = visualizer.get_current_errors("val/loss")
                val_acc = visualizer.get_current_errors("val/acc")
                filename = None
                if opt.criterion == "acc" and opt.mtl_mode != "ctc":
                    if val_acc < best_acc:
                        logging.info("val_acc {} > best_acc {}".format(val_acc, best_acc))
                        #opt.eps = adadelta_eps_decay(optimizer, opt.eps_decay)  # Epsilon constant for optimizer
                    else:
                        filename = "model.acc.best"
                    best_acc = max(best_acc, val_acc) 
                    logging.info("best_acc {}".format(best_acc))
                elif opt.criterion == "loss":
                    if val_loss > best_loss:
                        logging.info("val_loss {} > best_loss {}".format(val_loss, best_loss))
                        #opt.eps = adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename = "model.loss.best"
                    best_loss = min(val_loss, best_loss)
                    logging.info("best_loss {}".format(best_loss))
                state = {
                    "asr_state_dict": asr_model.state_dict(),
                    "opt": opt,
                    "epoch": epoch,
                    "iters": iters,
                    "eps": opt.eps,
                    "lr": opt.lr,
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "acc_report": acc_report,
                    "loss_report": loss_report,
                }
                save_checkpoint(state, opt.exp_path, filename=filename)

                visualizer.reset()


if __name__ == "__main__":
    train()
