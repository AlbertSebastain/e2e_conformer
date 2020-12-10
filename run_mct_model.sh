#!/bin/bash
#python3 ./asr_train_conf.py
#echo "confomer mct generated"
#bash ./run_conf_multi_recog.sh
#echo "conformer recog already done"
#bash ./joint_recog_att_multi.sh "enhance_recog" "e2e_mct_att" "True"
#bash ./joint_recog_conf_multi.sh "enhance_recog" "mct_conformer" "True"
#cp ./data_model/e2e_mct_att/model.acc.best ./data_model/asr_e2e_mct_retrain/model.e2e.att.best
#cp ./data_model/mct_conformer/model.acc.best ./data_model/asr_conformer_mct_retrain/model.e2e.conformer.best
#python3 ./asr_retrain.py
#echo "generated asr mct model"
#bash ./joint_recog_att_multi.sh "retrain" "asr_e2e_mct_retrain" "True"
#echo "decode done"
python3 ./asr_retrain_conf.py
echo "the conformer mct model is retrained"
bash ./joint_recog_conf_multi.sh "retrain" "asr_conformer_mct_retrain" "True"
echo "the tasks are fin"