#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration
stage=8        # start from 0 if you need to start from data preparation
#gpu=            # will be deprecated, please use ngpu
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=1      # verbose option
resume=${resume:=none}    # Resume the training from snapshot
# feature configuration
do_delta=false # true when using CNN
# network archtecture

# optimization related
opt=adadelta
epochs=20

# rnnlm related
model_unit=char
batchsize_lm=64
dropout_lm=0.5
input_unit_lm=256
hidden_unit_lm=650
lm_weight=0.2
fusion=${fusion:=none}
lmtype=rnnlm
# general config
train_set=train
train_dev=dev
recog_set="test"
test_set="test"
dev_set='dev'
checkpoints_dir="./checkpoints_test"
dataroot="/usr/home/shi/projects/data_aishell_test/data"
exp_root="/usr/home/shi/projects/e2e_speech_project/data_model_test"
dictroot="/usr/home/shi/projects/data_aishell_test/data/lang/phones"
data_noisedir="/usr/home/shi/projects/data_aishell/data/noise"
config_file_general="./config/general_config.yml"
dict=${dictroot}/${train_set}_units.txt
embed_init_file=${dictroot}/sgns
lmexpdir=${checkpoints_dir}/train_rnnlm
if [ ! -f config/path_config ]; then
    touch config/path_config
fi
echo "checkpoints_dir=${checkpoints_dir}" > config/path_config
echo "dataroot=${dataroot}" >> config/path_config
echo "exp_root=${exp_root}" >> config/path_config
echo "dictroot=${dictroot}" >> config/path_config
echo "lmexpdir=${lmexpdir}" >> config/path_config
echo "dict=${dict}" >> config/path_config
echo "embed_init_file=${embed_init_file}" >> config/path_config

config_file_asr=./config/att_spec.yml
train_file=${dataroot}/${train_set}
test_file=${dataroot}/${test_set}
dev_file=${dataroot}/${dev_set} 
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
# model name

enhance_name='enhance_fbank_train'
asr_name="aishell_asr_att_model"
joint_name="att_joint_train"
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

if [ ${stage} -le 1 ]; then
    echo "stage 1: feature preparation"
    feature_dir=("train" "test" "dev")
    cp -f ${train_file}/wav.scp ${train_file}/clean_wav.scp
    cp -f ${test_file}/wav.scp ${test_file}/clean_wav.scp
    cp -f ${dev_file}/wav.scp ${dev_file}/clean_wav.scp
    for((index=0;index<=2;index++)); do
        python3 data/prep_feats_mat.py ${dataroot} ${dataroot} 1 ${feature_dir[${index}]} ${data_noisedir}
    done
    echo "feature prepareation finish"
fi
# you can skip this and remove --rnnlm option in the recognition (stage 5)
nlsyms=${dictroot}/silence.txt
mkdir -p ${lmexpdir}
if [ ${stage} -le 2 ]; then
    echo "stage 2: LM Preparation"
    for x in train dev test; do
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${dataroot}/${x}/text.org) <(cut -f 2- -d" " ${dataroot}/${x}/text.org | tr -d " ") \
            > ${dataroot}/${x}/text
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text_char
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text_word
    #rm data/${x}/text.org
    done
    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 ${dataroot}/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    


    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1 -l ${nlsyms} --space "" ${dataroot}/train/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt

    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l  ${nlsyms} --space ""  ${dataroot}/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    echo "train language model"
    ${cuda_cmd} ${lmexpdir}/train.log \
        python3 lm_train.py \
        --ngpu 1 \
        --input-unit ${input_unit_lm} \
        --lm-type ${lmtype} \
        --unit ${hidden_unit_lm} \
        --dropout-rate ${dropout_lm} \
        --embed-init-file ${embed_init_file} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
    echo "LM finish"
        
fi


if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
    if [ ! -d ${exp_root}/${asr_name} ]; then
        mkdir -p ${exp_root}/${asr_name}
    fi
    if [ ! -d ${checkpoints_dir}/${asr_name} ]; then
        mkdir -p ${checkpoints_dir}/${asr_name}
    fi

    cp ${dataroot}/train/clean_feats.scp ${dataroot}/train/feats.scp
    cp ${dataroot}/test/clean_feats.scp ${dataroot}/test/feats.scp
    batchsize=30
    python3 asr_train.py \
    --dataroot $dataroot \
    --name $asr_name \
    --config_file ${config_file_asr} \
    --config_file ${config_file_general} \
    --batch-size ${batchsize} \
    --dict_dir ${dictroot} \
    --print_freq 500 \
    --validate_freq 3000 \
    --rnnlm ${lmexpdir}/rnnlm.model.best \
    --exp_path ${exp_root}/${asr_name} \
    --checkpoints_dir ${checkpoints_dir} \
    --epochs ${epochs} \
    --works_dir ${exp_root}/${asr_name}
    echo "att model finish"
fi
if [ ${stage} -le 4 ]; then
    echo "stage 4: recog clean data"
    if [ ! -d ${exp_root}/recog_multitype/clean_recog ]; then
        mkdir -p ${exp_root}/recog_multitype/clean_recog
    fi
    feat_dir=${dataroot}/test
    cp ${feat_dir}/clean_feats.scp ${feat_dir}/feats.scp
    if [ ! -f ${feat_dir}/clean_kaldi_feat_len.scp ]; then
        cp ${feat_dir}/kaldi_feat_len.scp ${feat_dir}/clean_kaldi_feat_len.scp
    else
        cp ${feat_dir}/clean_kaldi_feat_len.scp ${feat_dir}/kaldi_feat_len.scp
    fi
    bash ./recog_att.sh "clean_recog" "${asr_name}" "test"
    echo "recog clean data finish"
fi
if [ ${stage} -le 5 ]; then
    echo "stage 5: enhance model train"
    if [ ! -d ${exp_root}/${enhance_name} ]; then
        mkdir -p ${exp_root}/${enhance_name}
    fi
    if [ ! -d ${checkpoints_dir}/${enhance_name} ]; then
        mkdir -p ${checkpoints_dir}/${enhance_name}
    fi
    epochs=20
    batchsize=40
    print_freq=100
    validate_freq=3000
    python3 enhance_fbank_train.py \
        --dataroot ${dataroot} \
        --epochs ${epochs} \
        --config_file ${config_file_asr} \
        --config_file ${config_file_general} \
        --dict_dir ${dictroot} \
        --print_freq ${print_freq} \
        --batch-size ${batchsize} \
        --validate_freq ${validate_freq} \
        --works_dir ${exp_root}/${enhance_name} \
        --name ${enhance_name} \
        --checkpoints_dir ${checkpoints_dir} \
        --exp_path ${exp_root}/${enhance_name} \
        --works_dir ${exp_root}/${enhance_name}
    echo "enhance fbank model finish"
fi
if [ ${stage} -le 6 ]; then
    echo "stage 6: enhance model recog"
    feat_dir=${dataroot}/test
    if [ ! -f ${feat_dir}/clean_kaldi_feat_len.scp ]; then
        cp ${feat_dir}/kaldi_feat_len.scp ${feat_dir}/clean_kaldi_feat_len.scp
    fi
    sed 's/__mix0//g' ${feat_dir}/mix_feats.scp >${feat_dir}/feats.scp
    bash ./joint_recog_att.sh "enhance_recog" ${enhance_name} ${asr_name} "test"
    echo "recog mix data finish"
fi
if [ ${stage} -le 7 ]; then
    echo "stage 7: joint train"
    if [ ! -d ${exp_root}/${joint_name} ]; then
        mkdir -p ${exp_root}/${joint_name}
    fi
    if [ ! -d ${checkpoints_dir}/${joint_name} ]; then
        mkdir -p ${checkpoints_dir}/${joint_name}
    fi
    isGAN=true
    batch_size=30
    epochs=20
    validate_freq=3000
    cp ${exp_root}/${enhance_name}/model.loss.best ${exp_root}/${joint_name}/model.enhance.best
    cp ${exp_root}/${asr_name}/model.acc.best ${exp_root}/${joint_name}/model.att.best
    enhance_resume=model.enhance.best
    asr_resume=model.att.best
    config_file_joint_train=config/joint_train_config.yml
    python3 joint_train.py \
        --dataroot $dataroot \
        --name $joint_name \
        --dict_dir ${dictroot} \
        --config_file ${config_file_general} \
        --config_file ${config_file_asr} \
        --config_file ${config_file_joint_train} \
        --enhance_resume ${enhance_resume} \
        --asr_resume ${asr_resume} \
        --isGAN ${isGAN} \
        --exp_path ${exp_root}/${joint_name} \
        --batch-size ${batch_size} \
        --print_freq 500 \
        --validate_freq 3000 \
        --checkpoints_dir ${checkpoints_dir} \
        --rnnlm ${lmexpdir}/rnnlm.model.best \
        --epochs ${epochs} \
        --works_dir ${exp_root}/${joint_name} 
    echo "joint train finish"
fi
if [ ${stage} -eq 8 ]; then
    echo "stage 8: joint recog"
    feat_dir=${dataroot}/test
    mkdir -p ${checkpoints_dir}/recog_multitype/joint_train_recog
    sed 's/__mix0//g' ${feat_dir}/mix_feats.scp >${feat_dir}/feats.scp
    rm -f kaldi_feat_len.scp
    bash ./joint_recog_att.sh "joint_train_recog" ${enhance_name} ${joint_name} "test"
    echo "joint train recog finish"
fi

