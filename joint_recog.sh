#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
#stage=5        # start from 0 if you need to start from data preparation
#gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
verbose=1      # verbose option
# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
#etype=vggblstmp     # encoder architecture type
#elayers=8
#eunits=320
#eprojs=320
#subsample=1_2_2_1_1 # skip every n frame from input to nth layers
#subsample_type="skip"
# decoder related
#dlayers=1
#dunits=300
# attention related
#atype=location
#aact_func=softmax
#aconv_chans=10
#aconv_filts=100
#lsm_type="none"
#lsm_weight=0.0
#dropout_rate=0.0
# hybrid CTC/attention
mtlalpha=0.5


# optimization related
#opt=adadelta
#epochs=30
time_date=`date '+%Y%m%d_%H_%M'`
# rnnlm related
model_unit=char
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
decode_state=1

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

dataroot="/usr/home/shi/projects/data_aishell/data"
dictroot="/usr/home/shi/projects/data_aishell/data/lang/phones"
train_set=train
recog_set="test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}_units.txt
embed_init_file=${dictroot}/sgns
echo "dictionary: ${dictroot}"
nlsyms=${dictroot}/non_lang_syms.txt
lmexpdir=checkpoints_new/train_rnnlm
expdir="/usr/home/shi/projects/e2e_speech_project/data_model"
save_dir='/usr/home/shi/projects/e2e_speech_project/checkpoints_new'
nj=8

    for rtask in ${recog_set}; do
    ##(
        decode_dir=recog_joint_train_clean${rtask}
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
        #${decode_cmd} JOB=1:${nj} ${save_dir}/${decode_dir}/log/decode.JOB.log \
        #### use CPU for decoding  ##& 
        #${decode_cmd} JOB=1 ${save_dir}/${decode_dir}/log/decode.JOB.log \
        #${decode_cmd} JOB=1 ${save_dir}/${decode_dir}/log/decode.JOB.log \
        #if [ decode_state -eq 1 ]; then
            ${decode_cmd} JOB=1:${nj} ${save_dir}/${decode_dir}/log/decode.JOB.log \
                python3 joint_recog.py \
                --dataroot ${dataroot} \
                --dict_dir ${dictroot} \
                --name ${save_dir}/${decode_dir} \
                --model-unit $model_unit \
                --nj $nj \
                --gpu_ids 0 \
                --fbank_dim 80 \
                --nbest $nbest \
                --enhance_layers 3 \
                --joint_resume ${expdir}/joint_train/model.acc.best \
                --recog-dir ${sdata}/JOB  \
                --recog_data 'test' \
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
                #--enhance_resume ${expdir}/enhance_gan/model.loss.best \
                #--e2e_resume ${expdir}/asr_model/model.acc.best \
        #fi


            #--fstlm-path ${fst_path} \
            #--nn-char-map-file ${nn_char_map_file} \
          
        score_sclite.sh --nlsyms ${nlsyms} ${save_dir}/${decode_dir} ${dict}
        
        ##kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        ##rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &
    done
    ##wait
echo "Finished"