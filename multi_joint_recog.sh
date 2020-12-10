#!/bin/bash
######################
# the file is for multi recognitions 
# includes
# asr_model
# trained from   clean data
#                mct data
# recog set
#           cleantest
#           mix_match_test
#           mix_unmatch_test
#conformer
# trained from  clean data
#               mct data
# recog set
#               clean_test
#               mix_match_test
#               mix_unmatch_test
# enhance model
######################
#python3 ./joint_train.py 'asr_att_joint_train' 'mix_train_match' 'mix_dev_match'
python3 ./joint_train.py 'asr_att_mct_joint_train' 'mix_train_clean_match' 'mix_dev_clean_match'
echo "asr model finish"
bash ./joint_recog_att_multi.sh "joint_train_no_gan" "asr_att_mct_joint_train" "True"
python3 ./joint_train_conformer.py 'asr_conf_joint_train' 'mix_train_match' 'mix_dev_match'
python3 ./joint_train_conformer.py 'asr_conf_mct_joint_train' 'mix_train_clean_match' 'mix_dev_clean_match'
bash ./joint_recog_conf_multi.sh "joint_train_no_gan" "asr_conf_joint_train" "False" "asr_conf_mct_joint_train" "True"
#bash ./joint_recog_att_multi.sh
#echo "first e2e model recognition finish"
echo "conformer recognition finish"

