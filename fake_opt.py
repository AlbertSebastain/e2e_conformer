from tarfile import ENCODING
import torch

class Lm_train:
    def __init__(self):
        self.ngpu = 1
        self.input_unit = 256
        self.lm_type = "rnnlm"
        self.unit = 650
        self.dropout_rate = 0.5
        self.verbose = 1
        self.batchsize = 64
        self.outdir = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64"
        self.train_label = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/train.txt"
        self.valid_label = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/valid.txt"
        self.dict = "/usr/home/shi/projects/data_aishell/data/lang/phones/train_units.txt"
        self.embed_init_file = "/usr/home/shi/projects/data_aishell/data/lang/phones/sgns_wiki"
        self.seed = 1
        self.debugmode = 1
        self.bproplen = 35
        self.epoch = 25
        self.verbose = 1



class Asr_train:
    def __init__(self):
        self.feat_type = "kaldi_magspec"  #'kaldi_magspec'  # fbank not work
        self.delta_order = 0
        self.left_context_width = 0
        self.right_context_width = 0
        self.normalize_type = 1
        self.num_utt_cmvn = 20000
        self.model_unit = "char"
        self.exp_path = "/usr/home/shi/projects/e2e_speech_project/playground"

        self.dataroot = "/usr/home/shi/projects/data_aishell/data"
        self.name = "aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone"
        self.model_unit = "char"
        self.resume = None
        self.dropout_rate = 0.0
        self.etype = "blstmp"  # p代表projection
        self.elayers = 8
        self.eunits = 320
        self.eprojs = 320
        self.subsample = "1_2_2_1_1"
        self.subsample_type = "skip"
        self.dlayers = 1
        self.dunits = 300
        self.atype = "location"
        self.aact_fuc = "softmax"
        self.aconv_chans = 10
        self.aconv_filts = 100
        self.adim = 320
        self.mtlalpha = 0.5
        self.batch_size = 30
        self.maxlen_in = 800
        self.maxlen_out = 150
        self.opt_type = "adadelta"
        self.verbose = 1
        self.lmtype = "rnnlm"
        self.rnnlm = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/rnnlm.model.best"
        self.fusion = "None"
        self.epochs = 20

        self.checkpoints_dir = "./checkpoints"
        self.dict_dir = "/usr/home/shi/projects/data_aishell/data/lang/phones"
        self.num_workers = 4
        self.fbank_dim = 80  # 40
        self.lsm_type = ""
        self.lsm_weight = 0.0

        self.lr = 0.005
        self.eps = 1e-8  # default=1e-8
        self.iters = 0  # default=0
        self.start_epoch = 0  # default=0
        self.best_loss = float("inf")  # default=float('inf')
        self.best_acc = 0  # default=0
        self.sche_samp_start_iter = 5
        self.sche_samp_final_iter = 15
        self.sche_samp_final_rate = 0.6
        self.sche_samp_rate = 0.0

        self.shuffle_epoch = -1

        self.enhance_type = "blstm"
        self.fbank_opti_type = "frozen"
        self.num_utt_cmvn = 20000

        self.grad_clip = 5
        self.print_freq = 100
        self.validate_freq = 8000
        self.num_save_attention = 0.5
        self.criterion = "acc"
        self.mtl_mode = "mtl"
        self.eps_decay = 0.01
        self.gpu_ids = [1]
        self.lm_weight = 0.1
        self.enhance_resume = None
        self.enhance_layers = 3
        self.enhance_units = 128
        self.enhance_projs = 128
        self.enhance_nonlinear_type = 'sigmoid'
        self.enhance_loss_type = 'L2'
        self.enhance_opt_type = 'gan_fbank'
        self.enhance_dropout_rate = 0.0
        self.enhance_input_nc = 1
        self.enhance_output_nc = 1
        self.enhance_ngf = 64
        self.enhance_norm = 'batch'
        self.L1_loss_lambda = 1.0


class Enhance_base_train(Asr_train):
    def __init__(self):
        super(Enhance_base_train, self).__init__()


        self.feat_type = "kaldi_magspec"
        self.enhance_resume = None
        self.enhance_layers = 3
        self.enhance_units = 128
        self.enhance_projs = 128
        self.epochs = 1
        self.enhance_nonlinear_type = 'sigmoid'
        self.enhance_loss_type = 'L2'
        self.enhance_opt_type = 'gan_fbank'
        self.enhance_dropout_rate = 0.0
        self.enhance_input_nc = 1
        self.enhance_output_nc = 1
        self.enhance_ngf = 64
        self.enhance_norm = 'batch'
        self.L1_loss_lambda = 1.0
        self.name = "aishell"+"_enhancement"+"_"+str(self.enhance_layers)+"_"+str(self.etype)+"_"+str(self.enhance_units)+"_"+str(self.enhance_projs)+"_"+self.enhance_loss_type+"_"+str(self.enhance_dropout_rate)
class e2e_test(Asr_train):
    def __init__(self):
        super(e2e_test,self).__init__()
        self.recog_dir = "/usr/home/shi/projects/data_aishell/data/dev"
        self.enhance_dir = "/usr/home/shi/projects/data_aishell/data/dev"
        self.nj = 1
        self.nbest = 1
        self.beam_size = 1
        self.penalty = 0
        self.maxlenratio = 0.0
        self.minlenratio = 0.0
        self.ctc_weight = 1.2
        self.rescore = 3
        self.resume = "model.acc.best"
        self.works_dir = self.exp_path
        self.embed_init_file = "/usr/home/shi/projects/data_aishell/data/lang/phones/sgns"
        self.word_rnnlm = None
        #self.result_label = /data.JOB.json 
