import argparse
import os
from utils import utils
from distutils.util import strtobool
import yaml
import torch


class Base_conformer_Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # general configuration
        self.parser.add_argument('--works_dir', help='path to work', default='.')
        self.parser.add_argument("--feat_type",type=str,default="kaldi_magspec",help="feat_type")
        self.parser.add_argument("--delta_order", type=int, default=0, help="input delta-order")
        self.parser.add_argument('--dataroot', help='path (should have subfolders train, dev, test)')
        self.parser.add_argument('--left_context_width', type=int, default=0, help='input left_context_width-width')
        self.parser.add_argument('--right_context_width', type=int, default=0, help='input right_context_width')
        self.parser.add_argument('--normalize_type', type=int, default=1, help='normalize_type') 
        self.parser.add_argument('--num_utt_cmvn', type=int, help='the number of utterances for cmvn', default=20000)
        self.parser.add_argument('--dict_dir', default='/home/bliu/SRC/workspace/e2e/data/mix_aishell/lang_1char/', help='path to dict')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='', help='name of the experiment.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints1', help='models are saved here')  
        self.parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')   
        self.parser.add_argument('--enhance_resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--asr_resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')  
        self.parser.add_argument('--joint_resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')     
        self.parser.add_argument("--gan_resume",default=None,type=str,metavar="PATH")       
        self.parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
        self.parser.add_argument('--train_folder',default='train',type=str,help='name of train folder')
        self.parser.add_argument('--dev_folder',default='dev',type=str,help="name of dev folder")
        self.parser.add_argument('--exp_path',type=str,default = None,help = 'exp_dir')
        self.parser.add_argument("--mtlalpha",type = float, default = 0.0)
        # use yaml config
        self.parser.add_argument('--config_file',default=None,type=str,action = 'append',help="use yaml file to set arguments")
        # input features
        self.parser.add_argument("--transformer-init",type=str,default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal", 
                "kaiming_uniform",
                "kaiming_normal",
                ]
                ,help="how to initialize transformer parameters",)
        self.parser.add_argument("--transformer-input-layer",type=str,default="conv2d",
            choices=[
                "conv2d",
                "linear", 
                "embed"
                ]
                ,help="transformer input layer type",)
        self.parser.add_argument("--transformer-attn-dropout-rate",default=None,type=float,help="dropout in transformer attention. use --dropout-rate if None is set",)
        self.parser.add_argument("--transformer-lr",default=10.0,type=float,help="Initial value of learning rate",)
        self.parser.add_argument("--transformer-warmup-steps",default=25000,type=int,help="optimizer warmup steps",)
        self.parser.add_argument("--transformer-length-normalized-loss",default=True,type=strtobool,help="normalize loss by length",)
        self.parser.add_argument("--transformer-encoder-selfattn-layer-type",type=str,default="selfattn",
            choices=[
                "selfattn",
                "rel_selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
                ],
                help="transformer encoder self-attention layer type",)
        self.parser.add_argument("--transformer-decoder-selfattn-layer-type",type=str,default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
                ],
                help="transformer decoder self-attention layer type",)
        self.parser.add_argument("--wshare",default=4,type=int,help="Number of parameter shargin for lightweight convolution",)
        self.parser.add_argument("--ldconv-encoder-kernel-length",default="21_23_25_27_29_31_33_35_37_39_41_43",type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Encoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",)
        self.parser.add_argument("--ldconv-decoder-kernel-length",default="11_13_15_17_19_21",type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Decoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",)
        self.parser.add_argument("--ldconv-usebias",type=strtobool,default=False,help="use bias term in lightweight/dynamic convolution",)
        self.parser.add_argument("--dropout-rate",default=0.0,type=float,help="Dropout rate for the encoder",)
        self.parser.add_argument("--decoder_mode",default = None)
        self.parser.add_argument("--ctc_type",default = "warpctc",type=str)
        self.parser.add_argument("--report_cer",default = False, type = strtobool)
        self.parser.add_argument("--report_wer",default = False, type = strtobool)
        self.parser.add_argument("--elayers",default=4,type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",)
        self.parser.add_argument("--eunits","-u",default=300,type=int,help="Number of encoder hidden units",)
        # Attention
        self.parser.add_argument("--adim",default=320,type=int,help="Number of attention transformation dimensions",)
        self.parser.add_argument("--aheads",default=4,type=int,help="Number of heads for multi head attention",)
        # Decoder
        self.parser.add_argument("--dlayers", default=1, type=int, help="Number of decoder layers")
        self.parser.add_argument("--dunits", default=320, type=int, help="Number of decoder hidden units")
        self.parser.add_argument("--transformer-encoder-pos-enc-layer-type",type=str,default="abs_pos",choices=["abs_pos", "scaled_abs_pos", "rel_pos"],help="transformer encoder positional encoding layer type",)
        self.parser.add_argument("--transformer-encoder-activation-type",type=str,default="swish",choices=["relu", "hardtanh", "selu", "swish"],help="transformer encoder activation function type",)
        self.parser.add_argument("--macaron-style",default=False,type=strtobool,help="Whether to use macaron style for positionwise layer",)
    # CNN module
        self.parser.add_argument("--use-cnn-module",default=False,type=strtobool,help="Use convolution module or not",)
        self.parser.add_argument("--cnn-module-kernel",default=31,type=int,help="Kernel size of convolution module.",)
        
        # enhance model
        self.parser.add_argument('--enhance_type', default='blstm', type=str, 
                                  choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm'], 
                                  help='Type of enhance model architecture')
        self.parser.add_argument('--enhance_layers', default=3, type=int, help='Number of enhance model layers')
        self.parser.add_argument('--enhance_units', default=128, type=int, help='Number of enhance model hidden units')
        self.parser.add_argument('--enhance_projs', default=128, type=int, help='Number of enhance model projection units')
        self.parser.add_argument('--enhance_nonlinear_type', default='sigmoid', type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type')
        self.parser.add_argument('--enhance_loss_type', default='L2', type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type')
        self.parser.add_argument('--enhance_opt_type', default='gan_fbank', type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type')
        self.parser.add_argument('--enhance_dropout_rate', default=0.0, type=float, help='enhance_dropout_rate')
        self.parser.add_argument('--enhance_input_nc', default=1, type=int, help='enhance_input_nc')
        self.parser.add_argument('--enhance_output_nc', default=1, type=int, help='enhance_output_nc')
        self.parser.add_argument('--enhance_ngf', default=64, type=int, help='enhance_ngf')
        self.parser.add_argument('--enhance_norm', default='batch', type=str, help='enhance_norm')
        self.parser.add_argument('--L1_loss_lambda', default=1.0, type=float, help='L1_loss_lambda')
        self.parser.add_argument('--idim',default=257,type=int)
        self.parser.add_argument('--odim',default=257,type=int)
        self.parser.add_argument('--isGAN',action="store_true")
        
        # gan model
        self.parser.add_argument('--gan_loss_lambda', default=1.0, type=float, help='gan_loss_lambda')
        self.parser.add_argument('--netD_type', type=str, default='basic', help='selects model to use for netD [basic | n_layers | pixel]')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input channels: 1 for grayscale') 
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--norm_D', type=str, default='batch', help='instance normalization or batch normalization [batch | norm | none]')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--gan_resumne',type = str, default= None, help = 'the model of gan')
        
        # fbank model
        self.parser.add_argument('--fbank_dim', type=int, default=40, help='num_features of a frame')
        self.parser.add_argument('--fbank-opti-type', type=str, default='frozen', choices=['frozen', 'train'], help='fbank-opti-type')
        
        # model (parameter) related
        self.parser.add_argument('--sche-samp-rate', default=0.0, type=float, help='scheduled sampling rate')
        self.parser.add_argument('--sche-samp-final-rate', default=0.6, type=float, help='scheduled sampling final rate')
        self.parser.add_argument('--sche-samp-start-epoch', default=5, type=int, help='scheduled sampling start epoch')
        self.parser.add_argument('--sche-samp-final_epoch', default=15, type=int, help='scheduled sampling start epoch')
        
         # rnnlm related         
        self.parser.add_argument('--model-unit', type=str, default='char', choices=['char', 'word', 'syllable'], help='model_unit')
        self.parser.add_argument('--space-loss-weight', default=0.1, type=float, help='space_loss_weight.')
        self.parser.add_argument('--lmtype', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--rnnlm', type=str, default=None, help='RNNLM model file to read')
        self.parser.add_argument('--kenlm', type=str, default=None, help='KENLM model file to read')
        self.parser.add_argument('--word-rnnlm', type=str, default=None, help='Word RNNLM model file to read')
        self.parser.add_argument('--word-dict', type=str, default=None, help='Word list to read')
        self.parser.add_argument('--lm-weight', default=0.1, type=float, help='RNNLM weight.')
        
        # FSLSTMLM training configuration
        self.parser.add_argument('--fast_cell_size', type=int, default=400, help='fast_cell_size')
        self.parser.add_argument('--slow_cell_size', type=int, default=400, help='slow_cell_size')
        self.parser.add_argument('--fast_layers', type=int, default=2, help='fast_layers')
        self.parser.add_argument('--zoneout_keep_c', type=float, default=0.5, help='zoneout_c')
        self.parser.add_argument('--zoneout_keep_h', type=float, default=0.9, help='zoneout_h')
    
        # minibatch related
        self.parser.add_argument('--batch-size', '-b', default=30, type=int, help='Batch size')
        self.parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML', help='Batch size is reduced if the input sequence length > ML')
        self.parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML', help='Batch size is reduced if the output sequence length > ML')        
        self.parser.add_argument('--verbose', default=1, type=int, help='Verbose option')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        if self.opt.config_file != None:
            for config_file in  self.opt.config_file:
                with open(config_file,encoding = "utf-8") as f:
                    data_file = f.read()
                data = yaml.full_load(data_file)
                for key_d,val_d in data.items():
                    key_d = key_d.replace("--","")
                    key_d = key_d.replace("-","_")
                    setattr(self.opt,key_d,val_d)   
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if self.opt.mtlalpha == 1.0:
            self.opt.mtl_mode = 'ctc'
        elif self.opt.mtlalpha == 0.0:
            self.opt.mtl_mode = 'att'
        else:
            self.opt.mtl_mode = 'mtl'

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        #exp_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        utils.mkdirs(self.opt.exp_path)
        #self.opt.exp_path = self.opt.exp_path
        if self.opt.name != '':
            file_name = os.path.join(self.opt.checkpoints_dir,self.opt.name, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt

