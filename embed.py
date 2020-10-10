import torch
embed_init_file = "/usr/home/shi/projects/data_aishell/data/lang/phones/sgns"
print("oik")
word_dim = 0 
n = 0
with open(embed_init_file, 'r', encoding='utf-8') as fid:
    for line in fid:
        line_splits = line.strip().split()            
        word_dim = len(line_splits[1:])    
        #shape = (args.n_vocab, word_dim)
        #scale = 0.05
        #embed_vecs_init = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
#embed_weight_data.copy_(torch.from_numpy(embed_vecs_init))
print(word_dim)