#!/bin/bash
data_dir='/usr/home/shi/projects/data_aishell/data'
feats_dir=${data_dir}
data_type=('test' 'dev' 'train')
noise_repeat_num=1
for data in 'dev' 'train'
do
    echo "following ${data}:"
    python3 ./data/prep_feats_mat.py ${data_dir} ${feats_dir} ${noise_repeat_num} ${data}
    echo "finish ${data}"
done
