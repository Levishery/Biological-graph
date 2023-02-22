import sys
sys.path.append('/code/')

from biologicalgraphs.cnns.biological import edges, nodes
import os
import numpy as np
import numba 

import keras.backend as K
K.set_image_data_format('channels_first')

import argparse
import pytz  
from datetime import datetime
 

def get_cur_time():
    u = datetime.utcnow()
    u = u.replace(tzinfo=pytz.utc) #NOTE: it works only with a fixed utc offset
    t = datetime.now(tz=pytz.timezone('Asia/Shanghai'))
    cur_time = t.strftime("%Y%m%d%H%M")
    return cur_time


def print_params(parameters):
    print("---------network parameters------------")
    for k,v in parameters.items():
        print(k, v)

        
        
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--batchsize', type=int, default=40,
                   help='nvalidation_examples')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--epe', type=int, default=4000,
                   help='examples_per_epoch')
parser.add_argument('--nval', type=int, default=400,
                   help='nvalidation_examples')
args = parser.parse_args()

epochs = args.epochs
normalization = args.norm
examples_per_epoch = args.epe
nvalidation_examples = args.nval
batch_size = args.batchsize
initial_learning_rate = args.lr

if normalization:
    str_norm = 'yes'
else:
    str_norm = 'no'

    
# other paramethers 
subset = 'training'
width = (3,18,52,52)   # # The size of the input CNN.
network_radius=600   

threshold_volume = 10368000    
    
# cnn model parameters
parameters = {
'optimizer': 'nesterov',
'decay_rate': 5e-08,
'starting_epoch': 0,
'activation': 'LeakyReLU',
'batch_size': batch_size,
'filter_sizes': [16, 32, 64],
'depth': 3,
'initial_learning_rate': initial_learning_rate,
'weights': (1, 1),
'loss_function': 'mean_squared_error',
'betas': [0.99, 0.999],
'normalization': normalization,  # default False                      
 # Kasthuri is 20000,  Considering that our sample is only 5000 (1/4 of Kasthuri), so the epochs are only 500
'examples_per_epoch': examples_per_epoch,  
 # The following two were added by me
 # The original is 2000
'epochs': epochs,   
 # The original is 2000 Considering that our sample is only 5000 (1/4 of Kasthuri), so the epochs are only 500
'nvalidation_examples': nvalidation_examples
}   
# 
print_params(parameters)



cur_time = get_cur_time()

edge_model_prefix = '/data/onebear599/biologicalgraphs/biologicalgraphs/neuronseg/architectures/edges-600nm-3x18x18x52-ffn1-{}-bs{}-epe{}-nval{}-lr{}-norm-{}/edges'.format(cur_time, batch_size, examples_per_epoch, nvalidation_examples, initial_learning_rate, str_norm)

print(edge_model_prefix)


# ------------------------------------
# start trainningg
# width, network_radius主要都是为了读取训练样本所在目录， 体现在train.py的EdgeGenerator中
# model_prefix是为了存储网络的训练数据，作为一个目录保存
# parameters是cnn的参数
edges.train.Train(parameters=parameters, model_prefix=edge_model_prefix, width=width, radius=network_radius)
