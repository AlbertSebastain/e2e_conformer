#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# the files is for checking the lm model train units enhancement model e2e model are from kangyikai

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
#stage=5        # start from 0 if you need to start from data preparation
#gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
#resume=${resume:=none}    # Resume the training from snapshot
# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
subsample_type="skip"
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aact_func=softmax
aconv_chans=10
aconv_filts=100
lsm_type="none"
lsm_weight=0.0
dropout_rate=0.0
# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
#epochs=30
time_date=`date '+%Y%m%d_%H_%M'`
# rnnlm related
model_unit=char
batchsize_lm=64
dropout_lm=0.5
input_unit_lm=256
hidden_unit_lm=650
lm_weight=0.2
fusion=${fusion:=none}

# decoding parameter
lmtype=rnnlm
beam_size=12
nbest=12
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.1
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh
# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

dataroot="/usr/home/shi/projects/data_aishell/data"
##dictroot="/home/bliu/mywork/workspace/e2e/data/lang_1char/"
dictroot="/usr/home/shi/projects/data_aishell/data/lang/phones"
train_set=train
train_dev=dev
##recog_set="test_mix test_clean"
##recog_set="test_clean_small"
recog_set="kang_test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}_units_k.txt
embed_init_file=${dictroot}/sgns
echo "dictionary: ${dict}"
nlsyms=${dictroot}/non_lang_syms.txt
#lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
lmexpdir=checkpoints/train_rnnlm_k
mkdir -p ${lmexpdir}
name="aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_lr0.02_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone"
expdir="/usr/home/shi/projects/e2e_speech_project/playground"
save_dir='/usr/home/shi/projects/e2e_speech_project/checkpoints1'
#name=asr_clean_syllable_fbank80_drop0.2
#lmexpdir=checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64
#fst_path="/home/bliu/mywork/workspace/e2e/data/lang_word/LG_pushed_withsyms.fst"
#nn_char_map_file="/usr/home/shi/projects/data_aishell/dev/tex_cha"
nj=8
    for rtask in ${recog_set}; do
    ##(
        decode_dir=decode_joint_recog_checking_${rtask}_${lmtype}${lm_weight}_${time_date}
        #decode_dir=decode_joint_recog_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_${lmtype}${lm_weight}_10_15_20
        echo "${decode_dir}"
        feat_recog_dir=${dataroot}/${rtask}
        #./utils/fix_data_dir.sh ${feat_recog_dir}
        # split data        
        #splitjson.py --parts ${nj} ${feat_recog_dir}/data.json  --kenlm ${dictroot}/text.arpa \
        #cp -fp ${feat_recog_dir}/mix_feats.scp ${feat_recog_dir}/feats.scp
        sdata=${feat_recog_dir}/split$nj

        mkdir -p ${expdir}/${decode_dir}/log/
       

        #[[ -d $sdata && ${feat_recog_dir}/feats.scp -ot $sdata ]] || 
        utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
        echo $nj > ${expdir}/num_jobs
        #### use CPU for decoding  ##& 
        #${decode_cmd} JOB=1 ${save_dir}/${decode_dir}/log/decode.JOB.log \
        #${decode_cmd} JOB=1:${nj} ${save_dir}/${decode_dir}/log/decode.JOB.log \
        ${decode_cmd} JOB=1:${nj} ${save_dir}/${decode_dir}/log/decode.JOB.log \
           python3 joint_recog.py \
           --dataroot ${dataroot} \
           --dict_dir ${dictroot} \
           --name $name \
           --model-unit $model_unit \
           --nj $nj \
           --fbank_dim 80 \
           --gpu_ids 0 \
           --nbest $nbest \
           --enhance_layers 3 \
           --enhance_resume ${expdir}/base/model.loss.best \
           --e2e_resume ${expdir}/model.acc.best \
           --recog-dir ${sdata}/JOB \
           --result-label ${save_dir}/${decode_dir}/data.JOB.json \
           --beam-size ${beam_size} \
           --penalty ${penalty} \
           --maxlenratio ${maxlenratio} \
           --minlenratio ${minlenratio} \
           --ctc-weight ${ctc_weight} \
           --lmtype ${lmtype} \
           --verbose ${verbose} \
           --exp_path ${expdir} \
           --normalize_type 1 \
           --rnnlm ${lmexpdir}/rnnlm.model.best \
           --lm-weight ${lm_weight} \
           --embed-init-file ${embed_init_file} \
            #--fstlm-path ${fst_path} \
            #--nn-char-map-file ${nn_char_map_file} \


            #--fstlm-path ${fst_path} \
            #--nn-char-map-file ${nn_char_map_file} \
          
        score_sclite.sh --nlsyms ${nlsyms} ${save_dir}/${decode_dir} ${dict}
        
        ##kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        ##rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &
    done
    ##wait
echo "Finished"