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
######################
bash ./recog_att_multi.sh
echo "first e2e model recognition finish"
bash ./run_conf_multi_recog.sh
echo "conformer recognition finish"