path='/usr/home/shi/projects/data_aishell/data/mix_train_clean_match'
pathdev='/usr/home/shi/projects/data_aishell/data/mix_dev_clean_match'
sed 's/__mix0//g' ${path}/mix_feats.scp >${path}/feats.scp
sed 's/__mix0//g' ${pathdev}/mix_feats.scp >${pathdev}/feats.scp