#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.e2e_common import to_cuda, th_accuracy
from model.fsrnn import FSRNNLM
from data.lm_data_loader import ParallelSequentialIterator

class ClassifierWithState(nn.Module):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=F.cross_entropy,
                 accfun=th_accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))
        super(ClassifierWithState, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key
        self.predictor = predictor

    def forward(self, state, *args, **kwargs):
        """Computes the loss value for an input and label pair.az

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~torch.Tensor): Input minibatch.
            kwargs (dict of ~torch.Tensor): Input minibatch.

        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~torch.Tensor: Loss value.

        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None
        state, self.y = self.predictor(state, *args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
        return state, self.loss

    def predict(self, state, x):
        """Predict log probabilities for given state and input x using the predictor

        Returns:
            any type: new state
            TorchTensor: log probability vector

        """
        if hasattr(self.predictor, 'normalized') and self.predictor.normalized:
            return self.predictor(state, x)
        else:
            state, z = self.predictor(state, x)
            return state, F.log_softmax(z, dim=1).data


class RNNLM(nn.Module):

    def __init__(self, n_vocab, input_units, n_units, dropout_rate=0.5, embed_vecs_init=None):
        super(RNNLM, self).__init__()
        self.n_vocab = n_vocab
        self.n_units = n_units
        self.embed = torch.nn.Embedding(n_vocab, input_units)
        self.d0 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.LSTMCell(input_units, n_units)
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.LSTMCell(n_units, n_units)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.lo = torch.nn.Linear(n_units, n_vocab)

        # initialize parameters from uniform distribution
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)
        
        if embed_vecs_init is not None: 
            self.embed.weight.data.copy_(torch.from_numpy(embed_vecs_init))
            logging.info('load embed_vecs_init, shape ' + str(embed_vecs_init.shape[0]) + ' ' + str(embed_vecs_init.shape[1]))
        
    def zero_state(self, batchsize):
        return Variable(torch.zeros(batchsize, self.n_units)).float()

    def forward(self, state, x):
        if state is None:
            state = {
                'c1': to_cuda(self, self.zero_state(x.size(0))),
                'h1': to_cuda(self, self.zero_state(x.size(0))),
                'c2': to_cuda(self, self.zero_state(x.size(0))),
                'h2': to_cuda(self, self.zero_state(x.size(0)))
            }
        h0 = self.embed(x)
        h1, c1 = self.l1(self.d0(h0), (state['h1'], state['c1']))
        h2, c2 = self.l2(self.d1(h1), (state['h2'], state['c2']))
        y = self.lo(self.d2(h2))
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return state, y


def train(args):
    # display torch version
    logging.info('torch version = ' + torch.__version__)

    # seed setting
    nseed = args.seed
    torch.manual_seed(nseed)
    logging.info('torch seed = ' + str(nseed))

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda and cudnn availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    #with open(args.train_label, 'rb') as f:
        #train = np.array([args.char_list_dict[char]
                          #if char in args.char_list_dict else args.char_list_dict['<unk>']
                          #for char in f.readline().decode('utf-8').split()], dtype=np.int32)
    train = load_text(args.train_label,args.char_list_dict)
    # with open(args.valid_label, 'rb') as f:
    #     valid = np.array([args.char_list_dict[char]
    #                       if char in args.char_list_dict else args.char_list_dict['<unk>']
    #                       for char in f.readline().decode('utf-8').split()], dtype=np.int32)
    valid = load_text(args.valid_label,args.char_list_dict)

    logging.info('#vocab = ' + str(args.n_vocab))
    logging.info('#words in the training data = ' + str(len(train)))
    logging.info('#words in the validation data = ' + str(len(valid)))
    logging.info('#iterations per epoch = ' + str(len(train) // (args.batchsize * args.bproplen)))
    logging.info('#total iterations = ' + str(args.epoch * len(train) // (args.batchsize * args.bproplen)))

    # Create the dataset iterators
    train_iter = ParallelSequentialIterator(train, args.batchsize)
    valid_iter = ParallelSequentialIterator(valid, args.batchsize, repeat=False)
    
    if args.embed_init_file is not None:
        word_dim = 0 
        with open(args.embed_init_file, 'r', encoding='utf-8') as fid:
            for line in fid:
                line_splits = line.strip().split()               
                word_dim = len(line_splits[1:])
                break          
        shape = (args.n_vocab, word_dim)
        scale = 0.05
        embed_vecs_init = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
        args.input_unit = word_dim
        with open(args.embed_init_file, 'r', encoding='utf-8') as fid:
            try:
                for line in fid:
                    line_splits = line.strip().split()
                    char = line_splits[0]
                    if char in args.char_list_dict:
                        vector = np.array(map(float, line_splits[1:]), dtype='float32') 
                        index = args.char_list_dict[char]
                        embed_vecs_init[index] = vector
            except:
                pass                
    else:
        embed_vecs_init = None             
    
        
    # Prepare an RNNLM model
    if args.lm_type == 'rnnlm':
        rnn = RNNLM(args.n_vocab, args.input_unit, args.unit, args.dropout_rate, embed_vecs_init)
    elif args.lm_type == 'fsrnnlm':
        rnn = FSRNNLM(args.n_vocab, args.input_unit, args.fast_layers, args.fast_cell_size, args.slow_cell_size, 
                      args.zoneout_keep_h, args.zoneout_keep_c, args.dropout_rate, embed_vecs_init)
    print(rnn)
    model = ClassifierWithState(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    path = os.path.join(args.outdir, 'rnnlm.model.best')    
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))   
        logging.info('load model from ' + path)
         
    if args.ngpu > 1:
        logging.warn("currently, multi-gpu is not supported. use single gpu.")
    if args.ngpu > 0:
        # Make the specified GPU current
        gpu_id = 0
        model.cuda(gpu_id)
        
    # Set up an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    def evaluate(model, iter, bproplen=100):
        # Evaluation routine to be used for validation and test.
        # TODO(karita) use torch.no_grad here
        torch.set_grad_enabled(False)
        model.predictor.eval()
        state = None
        sum_perp = 0
        data_count = 0  
        
        while True:      
            try:
                batch = iter.__next__()
                batch = np.array(batch)
               
                x = torch.from_numpy(batch[:, 0]).long()
                t = torch.from_numpy(batch[:, 1]).long()
    
                if args.ngpu > 0:
                    x = x.cuda(gpu_id)
                    t = t.cuda(gpu_id)
                state, loss = model(state, x, t)
                sum_perp += loss.data
                if data_count % bproplen == 0:
                    # detach all states
                    for key in state.keys():
                        state[key] = state[key].detach()
                data_count += 1
            except:
                break
        # TODO(karita) use torch.no_grad here
        torch.set_grad_enabled(True)
        model.predictor.train()
        return np.exp(float(sum_perp) / data_count)
    
    if os.path.isfile(path):
        best_valid = evaluate(model, valid_iter)
    else:
        best_valid = 100000000
                           
    sum_perp = 0
    count = 0
    iteration = 0
    epoch_now = 0    
    state = None
    while train_iter.epoch < args.epoch:
        loss = 0
        iteration += 1
        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(args.bproplen):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            batch = np.array(batch)
            x = torch.from_numpy(batch[:, 0]).long()
            t = torch.from_numpy(batch[:, 1]).long()
            if args.ngpu > 0:
                x = x.cuda(gpu_id)
                t = t.cuda(gpu_id)
            # Compute the loss at this time step and accumulate it
            state, loss_batch = model(state, x, t)
            loss += loss_batch
            count += 1

        sum_perp += loss.data
        model.zero_grad()  # Clear the parameter gradients
        loss.backward()  # Backprop
        # detach all states
        for key in state.keys():
            state[key] = state[key].detach()
        
        nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
        optimizer.step()  # Update the parameters

        if iteration % 100 == 0:
            logging.info('iteration: ' + str(iteration))
            logging.info('training perplexity: ' + str(np.exp(float(sum_perp) / count)))
            sum_perp = 0
            count = 0

        if train_iter.epoch > epoch_now:
            valid_perp = evaluate(model, valid_iter)
            logging.info('epoch: ' + str(train_iter.epoch))
            logging.info('validation perplexity: ' + str(valid_perp))

            # Save the model and the optimizer
            logging.info('save the model')
            torch.save(model.state_dict(), args.outdir + '/rnnlm.model.' + str(epoch_now))
            logging.info('save the optimizer')
            torch.save(optimizer.state_dict(), args.outdir + '/rnnlm.state.' + str(epoch_now))

            if valid_perp < best_valid:
                dest = args.outdir + '/rnnlm.model.best'
                if os.path.lexists(dest):
                    os.remove(dest)
                os.symlink('rnnlm.model.' + str(epoch_now), dest)
                best_valid = valid_perp

            epoch_now = train_iter.epoch
def load_text(path_train_label,char_list_dict):
    train_set = []
    with open(path_train_label, 'rb') as f:
        line = f.readline().decode('utf-8').split('<space>')
        for char in line:
            char_new = char.replace(' ','')
            if char_new in char_list_dict:
                train_set.append(char_list_dict[char_new])
            else:
                chars = char.split()
                for c in chars:
                    if c in char_list_dict:
                        train_set.append(char_list_dict[c])
                    else:
                        train_set.append(char_list_dict['<unk>'])
    train = np.array(train_set,dtype = np.int32)
    return train
