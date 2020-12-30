#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

##########
# the file is for e2e attention model
# note the feats.scp in mix_test folder is the mix feats but delete the flag 'mix'
##########
# path config
if [ ! -f path_config ]; then
    source ./config/path_config
    save_type=$1
    model=$2
    recog_set=$3
else
    checkpoints_dir=$1
    dataroot=$2
    exp_root=$3
    dictroot=$4
    lmexpdir=$5
    dict=$6
    embed_init_file=$7
    save_type=$8
    model=$9
    recog_set=${10}
fi
# general configuration
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
verbose=1      # verbose option




# rnnlm related
model_unit=char
fusion=${fusion:=none}
config_file_decode=config/decode_config.yml

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

train_set=train
train_dev=dev

# you can skip this and remove --rnnlm option in the recognition (stage 5)
echo "dictionary: ${dict}"
nlsyms=${dictroot}/non_lang_syms.txt
save_dir="recog_multitype/${save_type}"
nj=8
for index_r in 0; do
##(
    rtask=${recog_set}
    decode_dir="recog_${model}_${rtask}"
    name=${save_dir}/${decode_dir}
    echo "the recog result is saved in"
    echo ${checkpoints_dir}/${name}  
    feat_recog_dir=${dataroot}/${rtask}
    expdir=${exp_root}/${model}
    feat_recog_dir=${dataroot}/${rtask}
    echo "${feat_recog_dir}"
    sdata=${feat_recog_dir}/split$nj
    utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
    echo $nj > ${expdir}/num_jobs
    ${decode_cmd} JOB=1:${nj} ${checkpoints_dir}/${save_dir}/${decode_dir}/log/decode.JOB.log \
        python3 asr_recog.py \
        --dataroot ${dataroot} \
        --dict_dir ${dictroot} \
        --name $name \
        --model-unit $model_unit \
        --nj $nj \
        --gpu_ids 0 \
        --resume ${expdir}/model.acc.best \
        --config_file ${config_file_decode} \
        --recog-dir ${sdata}/JOB/ \
        --checkpoints_dir ${checkpoints_dir} \
        --result-label ${checkpoints_dir}/${save_dir}/${decode_dir}/data.JOB.json \
        --verbose ${verbose} \
        --normalize_type 1 \
        --fbank_dim 80 \
        --rnnlm ${lmexpdir}/rnnlm.model.best \
        --embed-init-file ${embed_init_file} \
        --exp_path ${expdir} \
        --test_folder ${rtask}

        
    score_sclite.sh --nlsyms ${nlsyms} ${checkpoints_dir}/${save_dir}/${decode_dir} ${dict}
done
echo "Finished"
