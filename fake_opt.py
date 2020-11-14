from tarfile import ENCODING
import torch
import os

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
        self.exp_path = "/usr/home/shi/projects/e2e_speech_project/data_model/aishell_asr_mct_train"

        self.dataroot = "/usr/home/shi/projects/data_aishell/data"
        #self.name = "aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_lr0.02_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone"
        self.name = 'aishell_asr_mct_train'
        self.model_unit = "char"
        self.resume = None
        self.dropout_rate = 0.0
        self.etype = "blstmp"  # p代表projection
        self.elayers = 4
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
        self.rnnlm = "checkpoints_new/train_rnnlm/rnnlm.model.best"
        self.fusion = "None"
        self.epochs = 20

        self.checkpoints_dir = "./checkpoints_new"
        self.dict_dir = "/usr/home/shi/projects/data_aishell/data/lang/phones"
        self.num_workers = 8
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
        self.print_freq = 1000
        self.validate_freq = 3000
        self.num_save_attention = 0.5
        self.criterion = "acc"
        self.mtl_mode = "mtl"
        self.eps_decay = 0.01
        self.gpu_ids = [0]
        self.lm_weight = 0.2
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
        self.checkpoints_dir = './checkpoints_new'
        self.name = 'aishell_e2e_mct_att_train'
        self.exp_path = './data_model/e2e_mct_att'
        self.MCT  = True
        self.train_folder = 'mix_train_clean_match'
        self.dev_folder = 'mix_dev_clean_match'
        # self.validate_freq = 1

class asr_conf(Asr_train):
    def __init__(self):
        super(asr_conf,self).__init__()
        self.eps = 1e-9
        self.beta1 = 0.9
        self.transformer_init = 'pytorch'
        self.transformer_input_layer = 'conv2d'
        self.transformer_attn_dropout_rate = 0.0
        self.transformer_lr = 1
        self.transformer_warmup_steps = 25000
        self.transformer_length_normalized_loss = False
        self.transformer_encoder_selfattn_layer_type = 'rel_selfattn'
        self.transformer_decoder_selfattn_layer_type = 'selfattn'
        self.wshare = 4
        self.ldconv_encoder_kernel_length = '21_23_25_27_29_31_33_35_37_39_41_43'
        self.ldconv_decoder_kernel_length = '11_13_15_17_19_21'
        self.ldconv_usebias = False
        self.dropout_rate = 0.0
        self.elayers = 12
        self.eunits = 2048
        self.adim = 256
        self.aheads = 4
        self.dlayers = 6
        self.dunits = 2048
        self.transformer_encoder_pos_enc_layer_type = 'rel_pos'
        self.macaron_style = True
        self.use_cnn_module = True
        self.cnn_module_kernel = 31
        self.transformer_encoder_activation_type = 'swish'
        self.dropout_rate = 0.1
        self.report_cer = False
        self.ctc_type = 'warpctc'
        self.report_wer = False
        self.lr = 0.02
        self.epochs = 50
        self.print_freq = 500
        self.validate_freq = 3000
        self.mtlalpha = 0.3
        self.lsm_weight = 0.1
        self.opt_type = 'noam'
        self.accum_grad = 2
        self.grad_clip = 5
        self.dropout_rate = 0.1
        self.exp_path = './data_model/mct_conformer'
        self.works_dir = self.exp_path
        self.decoder_mode = None
        self.batch_size = 15
        self.maxlen_in = 512
        self.maxlen_out = 150
        self.resume = 'model.acc.best'
        self.name = 'asr_e2e_mct_conformtrain_test'
        self.MCT = True
class asr_recog_conf(asr_conf):
    def __init__(self):
        super(asr_recog_conf,self).__init__()
        self.recog_dir = '/usr/home/shi/projects/data_aishell/data/test'
        self.nj = 1
        self.nbest = 12
        self.beam_size = 10
        self.maxlenratio = 0.0
        self.minlenratio = 0.0
        self.ctc_weight = 0.5
        self.resume = 'model.acc.best'
        self.works_dir = self.exp_path
        self.embed_init_file = '/usr/home/shi/projects/data_aishell/data/lang/phones/sgns'
        self.word_rnnlm = None
        self.name = 'recog_conf_testff'
        self.rnnlm = '/usr/home/shi/projects/e2e_speech_project/checkpoints_new/train_rnnlm/rnnlm.model.best'
        self.penalty = 0
        self.lm_weight = 0.7
        self.ngram_weight = 0.3



class Enhance_base_train(Asr_train):
    def __init__(self):
        super(Enhance_base_train, self).__init__()


        self.feat_type = "kaldi_magspec"
        self.enhance_layers = 3
        self.enhance_units = 128
        self.enhance_projs = 128
        self.epochs = 30
        self.print_freq = 100
        self.num_saved_specgram = 3
        self.enhance_nonlinear_type = 'sigmoid'
        self.enhance_loss_type = 'L2'
        self.enhance_opt_type = 'gan_fbank'
        self.enhance_dropout_rate = 0.0
        self.enhance_input_nc = 1
        self.enhance_output_nc = 1
        self.enhance_ngf = 64
        self.enhance_norm = 'batch'
        self.L1_loss_lambda = 1.0
        self.batch_size = 40
        self.etype = 'blstmp'
        self.idim = 257
        self.odim = self.idim

        self.exp_path = "/usr/home/shi/projects/e2e_speech_project/data_model/enhance_base"
        self.works_dir = self.exp_path
        self.enhance_resume = None
        #self.name = "aishell"+"_enhancement"+"_"+str(self.enhance_layers)+"_"+str(self.etype)+"_"+str(self.enhance_units)+"_"+str(self.enhance_projs)+"_"+self.enhance_loss_type+"_"+str(self.enhance_dropout_rate)
        self.name = "enhance_base"
class e2e_test(Asr_train):
    def __init__(self):
        super(e2e_test,self).__init__()
        self.recog_dir = "/usr/home/shi/projects/data_aishell/data/test"
        self.enhance_dir = "/usr/home/shi/projects/data_aishell/data/test"
        self.nj = 1
        self.nbest = 12
        self.beam_size = 12
        self.penalty = 0
        self.maxlenratio = 0.0
        self.minlenratio = 0.0
        self.ctc_weight = 0.1
        #self.rescore = 3
        self.resume = "model.acc.best"
        self.works_dir = self.exp_path
        self.embed_init_file = "/usr/home/shi/projects/data_aishell/data/lang/phones/sgns"
        self.word_rnnlm = None
        #self.result_label = /data.JOB.json 
class joint_recog(e2e_test):
    def __init__(self):
        super(joint_recog,self).__init__()
        self.exp_path = "/usr/home/shi/projects/e2e_speech_project/data_model"
        self.enhance_resume = '/usr/home/shi/projects/e2e_speech_project/data_model/enhance_base/model.loss.best'
        self.e2e_resume = '/usr/home/shi/projects/e2e_speech_project/data_model/asr_model/model.acc.best'
        self.works_dir = '.'
        self.verbose = 1
        self.recog_data = 'test'
        self.recog_dir = '/usr/home/shi/projects/data_aishell/data/mix_test'
        self.name = 'joint_decode_with_no_update_cmvn_for_try'
        #self.rnnlm = "checkpoints1/train_rnnlm_k/rnnlm.model.best"
class Enhance_base_fbank_train(Enhance_base_train):
    def __init__(self):
        super(Enhance_base_fbank_train,self).__init__()
        #self.name = "aishell"+"_enhancement"+"_fbank_train"+str(self.enhance_layers)+"_"+str(self.etype)+"_"+str(self.enhance_units)+"_"+str(self.enhance_projs)+"_"+self.enhance_loss_type+"_"+str(self.enhance_dropout_rate)
        self.name = 'enhance_fbank_no_fbank'
        self.exp_path = "/usr/home/shi/projects/e2e_speech_project/data_model/base_fbank_train_nofbank"
        self.works_dir = self.exp_path
        self.num_saved_specgram = 6
        self.enhance_resume ='latest'
        self.enhance_resume = 'latest'
        self.epochs = 21
        self.validate_freq = 3000
class joint_recog_check(joint_recog):
    def __init__(self):
        super(joint_recog_check,self).__init__()
        self.rnnlm = 'checkpoints1/train_rnnlm/rnnlm.model.best'
        self.e2e_resume = '/usr/home/shi/projects/e2e_speech_project/playground/model.acc.best'
        self.enahnce_resume = '/usr/home/shi/projects/e2e_speech_project/playground/base/model.loss.best'
        self.recog_dir = "/usr/home/shi/projects/data_aishell/data/dev_mix"
        self.recog_data = 'dev'
        args = vars(self)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
class Enhance_gan_train(Enhance_base_train):
    def __init__(self):
        super(Enhance_gan_train,self).__init__()
        self.name = 'aishell_enhance_gan'
        self.exp_path = '/usr/home/shi/projects/e2e_speech_project/data_model/enhance_gan'
        self.works_dir = self.exp_path
        self.num_saved_specgram = 6
        self.enhance_resume = 'latest'
        self.epochs = 20
        self.validate_freq = 3000
        self.ndf = 32
        self.norm_D = 'batch'
        self.no_lsgan = False
        self.input_nc = 1
        self.n_layers_D = 3
        #self.enhance_resume = None
        self.netD_type = 'n_layers'
        self.gan_loss_lambda = 1.0
class joint_train(Enhance_base_train):
    def __init__(self):
        super(joint_train,self).__init__()
        self.name = 'aishell_joint_train'
        self. exp_path = '/usr/home/shi/projects/e2e_speech_project/data_model/joint_train'
        self.works_dir = self.exp_path
        self.num_save_attention = 0.5
        self.enhance_resume = 'model.fbank.best'
        self.asr_resume = 'model.e2e.best'
        self.joint_resume = None
        self.epochs = 20
        self.validate_freq = 3000
        self.ndf = 32
        self.norm_D = 'batch'
        self.no_lsgan = False
        self.input_nc = 1
        self.n_layers_D = 3
        self.netD_type = 'n_layers'
        self.gan_loss_lambda = 1.0
        self.mtlalpha = 0.5
        self.enhance_loss_lambda = 1.0
        self.isGAN = True
        self.gan_resume = 'model.gan.best'

