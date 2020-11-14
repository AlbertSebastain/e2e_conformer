#!/bin/bash
data_dir='/usr/home/shi/projects/data_aishell/data_differ'
data_type=('test' 'dev' 'train')
noise_repeat_num=1
dblist=(-5 5 10)
noise_l='/usr/home/shi/projects/data_aishell/data/noise'

dbdir=('datadb_n5' 'datadb_p5' 'datadb_p10')
noisedir=('data_w' 'data_f' 'data_d')
noise_type=('white.mat' 'factory1.mat' 'destroyerengine.mat')
for i in 0 1 2
do
    db=${dblist[i]}
    for j in 0 1 2
    do
        noise=${noise_type[j]}
        for k in 'train' 'dev' 'test'
        do
            dir_name=${data_dir}/${dbdir[i]}/${noisedir[j]}
            echo ${dir_name}/${k}
            feats_dir=${dir_name}
            noise_path=${noise_l}/${noise}
            python3 ./data/prepare_feats_specific.py ${dir_name} ${feats_dir} ${noise_repeat_num} ${k} ${db} ${noise_path}
    #echo "following ${data}:"
    #python3 ./data/prep_feats_mat.py ${data_dir} ${feats_dir} ${noise_repeat_num} ${data}
    #echo "finish ${data}"
        done
    done
done
