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
import json
import logging
import torch
import fake_opt

from options.test_options import TestOptions
from model.feat_model import FFTModel, FbankModel
from model.e2e_model import E2E
from model import lm, extlm, fsrnn
from model.fstlm import NgramFstLM
from data.data_loader import SequentialDataset, SequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils
from char_error_rate import char_error
opt = TestOptions().parse()
if opt.recog_dir == '':
    opt = fake_opt.e2e_test()
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)  

# logging info
if opt.verbose == 1:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
elif opt.verbose == 2:
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
else:
    logging.basicConfig(
        level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    logging.warning("Skip DEBUG/INFO messages")
    
# data
def str2bool(stri):
    return True if stri.lower() == 'true' else False
if type(opt.MCT) == str:
    opt.MCT = str2bool(opt.MCT)
logging.info("Building dataset.")
recog_dataset = SequentialDataset(opt, opt.recog_dir, os.path.join(opt.dict_dir, 'train_units.txt'),type_data = opt.test_folder,mct = opt.MCT) 
recog_loader = SequentialDataLoader(recog_dataset, batch_size=1, num_workers=opt.num_workers, shuffle=False)
opt.idim = recog_dataset.get_feat_size()
opt.odim = recog_dataset.get_num_classes()
opt.char_list = recog_dataset.get_char_list()
opt.labeldist = recog_dataset.get_labeldist()
print('#input dims : ' + str(opt.idim))
print('#output dims: ' + str(opt.odim))
logging.info("Dataset ready!")

                                              
def main():
    
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if not os.path.isfile(model_path):
            raise Exception("no checkpoint found at {}".format(model_path))
        
        package = torch.load(model_path, map_location=lambda storage, loc: storage)        
        feat_model = FbankModel.load_model(model_path, 'fbank_state_dict', opt) 
        asr_model = E2E.load_model(model_path, 'asr_state_dict', opt) 
        logging.info('Loading model {}'.format(model_path))
    else:
        raise Exception("no checkpoint found at {}".format(opt.resume))
  
    def cpu_loader(storage, location):
        return storage
        
    if opt.lmtype == 'rnnlm':         
        # read rnnlm
        if opt.rnnlm:
            if opt.embed_init_file is not None:
                word_dim = 0 
                with open(opt.embed_init_file, 'r', encoding='utf-8') as fid:
                    for line in fid:
                        line_splits = line.strip().split()               
                        word_dim = len(line_splits[1:])
                        break          
                shape = (len(opt.char_list), word_dim)
                scale = 0.05
                embed_vecs_init = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
                #input_unit = word_dim
                #with open(opt.embed_init_file, 'r', encoding='utf-8') as fid:
                        #try:
                            #for line in fid:
                            #line_splits = line.strip().split()
                            #char = line_splits[0]
                            #if char in opt.char_list_dict:
                                #vector = np.array(map(float, line_splits[1:]), dtype='float32') 
                                #index = args.char_list_dict[char]
                                #embed_vecs_init[index] = vector
                        #except:
                            #pass                
            else:
                embed_vecs_init = None 
            rnnlm = lm.ClassifierWithState(
                lm.RNNLM(len(opt.char_list), word_dim, 650,embed_vecs_init = embed_vecs_init))
            rnnlm.load_state_dict(torch.load(opt.rnnlm, map_location=cpu_loader))
            if len(opt.gpu_ids) > 0: 
                rnnlm = rnnlm.cuda()     
            print('load RNNLM from {}'.format(opt.rnnlm))
            rnnlm.eval()
        else:
            rnnlm = None
        
        if opt.word_rnnlm:
            if not opt.word_dict:
                logging.error('word dictionary file is not specified for the word RNNLM.')
                sys.exit(1)

            word_dict = load_labeldict(opt.word_dict)
            char_dict = {x: i for i, x in enumerate(opt.char_list)}
            word_rnnlm = lm.ClassifierWithState(lm.RNNLM(len(word_dict), 650))
            word_rnnlm.load_state_dict(torch.load(opt.word_rnnlm, map_location=cpu_loader))
            word_rnnlm.eval()

            if rnnlm is not None:
                rnnlm = lm.ClassifierWithState(
                    extlm.MultiLevelLM(word_rnnlm.predictor,
                                               rnnlm.predictor, word_dict, char_dict))
            else:
                rnnlm = lm.ClassifierWithState(
                    extlm.LookAheadWordLM(word_rnnlm.predictor,
                                                  word_dict, char_dict))
        fstlm = None
        
    elif opt.lmtype == 'fsrnnlm':
        if opt.rnnlm:
            rnnlm = lm.ClassifierWithState(
                          fsrnn.FSRNNLM(len(opt.char_list), 300, opt.fast_layers, opt.fast_cell_size, 
                          opt.slow_cell_size, opt.zoneout_keep_h, opt.zoneout_keep_c))
            rnnlm.load_state_dict(torch.load(opt.rnnlm, map_location=cpu_loader))
            if len(opt.gpu_ids) > 0: 
                rnnlm = rnnlm.cuda()   
            print('load fsrnn from {}'.format(opt.rnnlm))
            rnnlm.eval()
        else:
            rnnlm = None
            print('not load fsrnn from {}'.format(opt.rnnlm))
        fstlm = None
                                                  
    elif opt.lmtype == 'fstlm':   
        if opt.fstlm_path: 
            fstlm = NgramFstLM(opt.fstlm_path, opt.nn_char_map_file, 20)
        else:
            fstlm = None 
        rnnlm = None
    else:
        rnnlm = None
        fstlm = None
    if opt.MCT == True:
        fbank_cmvn_file = os.path.join(opt.exp_path, 'fbank_mct_cmvn.npy')
    else:
        fbank_cmvn_file = os.path.join(opt.exp_path, 'fbank_cmvn.npy')
    if os.path.exists(fbank_cmvn_file):
        fbank_cmvn = np.load(fbank_cmvn_file)
        fbank_cmvn = torch.FloatTensor(fbank_cmvn)
    else:
        raise Exception("no found at {}".format(fbank_cmvn_file))
            
    torch.set_grad_enabled(False)
    new_json = {}
    for i, (data) in enumerate(recog_loader, start=0):
        utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
        name = utt_ids[0]
        print(name)
        if opt.feat_type == 'kaldi_fbank':
            feats = (inputs + fbank_cmvn[0, :]) * fbank_cmvn[1, :]  
        else:
            feats = feat_model(inputs, fbank_cmvn)
            #c_error = char_error(opt.rescore)
        #for num in range(0,opt.rescore):        
            nbest_hyps = asr_model.recognize(feats, opt, opt.char_list, rnnlm=rnnlm, fstlm=fstlm)
            # get 1best and remove sos
            y_hat = nbest_hyps[0]['yseq'][1:]
            ##y_true = map(int, targets[0].split())
            y_true = targets
            
            # print out decoding result
            seq_hat = [opt.char_list[int(idx)] for idx in y_hat]
            seq_true = [opt.char_list[int(idx)] for idx in y_true]
            seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
            seq_true_text = "".join(seq_true).replace('<space>', ' ')
            #c_error.error_rate_compute(seq_true,seq_hat,spk_ids[0],utt_ids[0])

        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)
        #logging.info("char error rate for the utterance %d " + c_error.uttid_er_rate[utt_ids[0]])
        # copy old json info
        new_json[name] = dict()
        new_json[name]['utt2spk'] = spk_ids[0]

        # added recognition results to json
        logging.debug("dump token id")
        out_dic = dict()
        out_dic['name'] = 'target1'
        out_dic['text'] = seq_true_text
        out_dic['token'] = " ".join(seq_true)
        out_dic['tokenid'] = " ".join([str(int(idx)) for idx in y_true])

        # TODO(karita) make consistent to chainer as idx[0] not idx
        out_dic['rec_tokenid'] = " ".join([str(int(idx)) for idx in y_hat])
        #logger.debug("dump token")
        out_dic['rec_token'] = " ".join(seq_hat)
        #logger.debug("dump text")
        out_dic['rec_text'] = seq_hat_text

        new_json[name]['output'] = [out_dic]
        # TODO(nelson): Modify this part when saving more than 1 hyp is enabled
        # add n-best recognition results with scores
        if opt.beam_size > 1 and len(nbest_hyps) > 1:
            for i, hyp in enumerate(nbest_hyps):
                y_hat = hyp['yseq'][1:]
                seq_hat = [opt.char_list[int(idx)] for idx in y_hat]
                seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
                new_json[name]['rec_tokenid' + '[' + '{:05d}'.format(i) + ']'] = " ".join([str(idx) for idx in y_hat])
                new_json[name]['rec_token' + '[' + '{:05d}'.format(i) + ']'] = " ".join(seq_hat)
                new_json[name]['rec_text' + '[' + '{:05d}'.format(i) + ']'] = seq_hat_text
                new_json[name]['score' + '[' + '{:05d}'.format(i) + ']'] = float(hyp['score'])
    # TODO(watanabe) fix character coding problems when saving it
    with open(opt.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4, sort_keys=True).encode('utf_8'))
         
      
if __name__ == '__main__':
    main()
