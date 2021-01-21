# joint training for ASR rnn_attention based model and conformer framework
The repository implements an ASR framework. We proposed a joint training network that contains an enhancement model a ASR End-to-End model and a discriminant model. For comparson purpose we implement two ASR End-to-End models, RNN-Attention based model and conformer model. For better robost performance, we add a discriminant model to guide the enhancement networks towards true signal. 
## Requirements
python 3,5 Pytorch 0.4.0
### Data
For evaluation the performance, we use  AISHELL-1 to train the model, which is a mandrian corpus dataset. You can download the AISHELL-1 online.  
For training the enhancement network, we use NOISE-92 as the background noise.  
You can also use your own dataset. In the dataset there must be three parts train set, develop set and test set.   
## Framework
The input data of the enhancement model is 257 dimension STFT feature. The enhancement network will transform the data to a mask which has the size as input data. The estimated clean signal is computed as $\hat{M}\otimes X$, where $\hat{M}$ is estimated mask and $X$ is the speech in the noise environment.  
In the joint training, the discriminant network can guide the enhancement network training towards true clean signal. The output of the discriminant network is a probability of the input data is true clean signal or estiamted clean signal.  
## run the framework
you can run `run_att_model.sh` or `run_conformer_model` to train and test the model.  

