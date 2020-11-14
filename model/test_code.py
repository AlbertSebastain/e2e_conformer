import numpy as np
class lmt:
    def __init__(self):
        #self.train_label = '/usr/home/shi/projects/data_aishell/data/train/text'
        self.dict = '/usr/home/shi/projects/data_aishell/data/lang/phones/train_units.txt'
        self.char_list_dict = {}
        self.train_label = '/usr/home/shi/projects/e2e_speech_project/checkpoints1/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/train.txt'
args = lmt()
train_set = []
with open(args.dict, 'rb') as f:
    dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')
    args.char_list_dict = {x: i for i, x in enumerate(char_list)}
    args.n_vocab = len(char_list)
with open(args.train_label, 'rb') as f:
    line = f.readline().decode('utf-8').split('<space>')
    for char in line:
        char_new = char.replace(' ','')
        if char_new in args.char_list_dict:
            train_set.append(args.char_list_dict[char_new])
        else:
            chars = char.split()
            for c in chars:
                if c in args.char_list_dict:
                    train_set.append(args.char_list_dict[c])
                else:
                    train_set.append(args.char_list_dict['<unk>'])
with open(args.train_label, 'rb') as f:
   train = np.array([args.char_list_dict[char]
                   if char in args.char_list_dict else args.char_list_dict['<unk>']
                   for char in f.readline().decode('utf-8').split()], dtype=np.int32)
            
train1 = np.array(train_set, dtype=np.int32)
print(train)