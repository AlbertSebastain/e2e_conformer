'''
build a mix dataset, mkdir data mix_test mix_dev copy from test dev
use mix_feats.scp as feats.scp and delete __mix
'''
import os
from shutil import copyfile
import shutil
path = '/usr/home/shi/projects/data_aishell/data'
type_data = ['test','dev']
for data in type_data:
    path_data = os.path.join(path,data)
    mix_data_name = 'mix_'+data
    if os.path.isdir(os.path.join(path,mix_data_name)):
        shutil.rmtree(os.path.join(path,mix_data_name))
    os.mkdir(os.path.join(path,mix_data_name))
    for (dirpath,dirname,file_names) in os.walk(path_data):
        filename = [name for name in file_names if file_names != 'feats.scp']
        break
    for file_data in filename:
        copyfile(os.path.join(path_data,file_data), os.path.join(path,mix_data_name,file_data))
    wr_file = open(os.path.join(path,mix_data_name,'feats.scp'),'w')
    with open(os.path.join(path,mix_data_name,'mix_feats.scp'),'r') as f:
        lines = f.readlines()
        for line in lines:
            newline = line.replace('__mix0','')
            wr_file.write(newline)
    wr_file.close()
