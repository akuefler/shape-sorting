from shapesorting import *

from autoencoder_lib import AutoEncoder
from sandbox.util import Saver

import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name',type=str,default='shape_encoder')
parser.add_argument('--data_name',type=str,default='similarity-00000')

# encoding models
parser.add_argument('--autoencoder_name',type=str,default='shape_encoder-00005-00001')
parser.add_argument('--dqnencoder_name',type=str,default='shape_encoder_dqn-00000')

args = parser.parse_args()

encoding_saver = Saver(args.encoding_name, path=DATADIR)
similarity_saver = Saver(args.data_name,path= DATADIR,overwrite=True)
autoencoder_saver = Saver(args.autoencoder_name,path=DATADIR,overwrite=True)
dqnencoder_saver = Saver(args.dqnencoder_name,path=DATADIR,overwrite=True)

X1 = similarity_saver.load_value(0, "X1")
X2 = similarity_saver.load_value(0, "X2")
Y = similarity_saver.load_value(0, "Y")
N = X1.shape[0]

autoencoder_weights = autoencoder_saver.load_dictionary(0,'encoder')
dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

#convert = {'l1_w':'l1_W:0',
           #'l2_w':'l2_W:0',
           #'l3_w':'l3_W:0',
           #'l4_w':'l4_W:0',
           #'q_w':'q_W:0',
           #'l1_b':'l1_b:0',
           #'l2_b':'l2_b:0',
           #'l3_b':'l3_b:0',
           #'l4_b':'l4_b:0',
           #'q_b':'q_b:0',          
           #}

#convert2 = {v:k for k, v in convert.iteritems()}

#ENCODER.set_weights([autoencoder_weights[convert2[w.name]] for w in ENCODER.weights])

baseline_encoder = AutoEncoder(20)
reinforc_encoder = AutoEncoder(20)

with tf.Session() as sess:
    baseline_encoder.load_weights(autoencoder_weights)
    reinforc_encoder.load_weights(dqnencoder_weights)

    for layer in baseline_encoder.layers.keys():
        bZ1 = []
        bZ2 = []
        rZ1 = []
        rZ2 = []
        for i in range(0,N,args.batch_size):
            X1_batch = X1[i:i+args.batch_size]
            X2_batch = X2[i:i+args.batch_size]
            
            bz1 = baseline_encoder.encode(X1_batch, layer=layer)
            bz2 = baseline_encoder.encode(X2_batch, layer=layer)
            rz1 = reinforc_encoder.encode(X1_batch, layer=layer)
            rz2 = reinforc_encoder.encode(X2_batch, layer=layer)
            
            bZ1.append(bz1)
            bZ2.append(bz2)
            rZ1.append(rz1)
            rZ2.append(rz2)
            
        D = {'bZ1':np.concatenate(bZ1,axis=0),
             'bZ2':np.concatenate(bZ2,axis=0),
             'rZ1':np.concatenate(bZ1,axis=0),
             'rZ2':np.concatenate(bZ2,axis=0)}
        encoding_saver.save_dict(0, D, name="{}_encodings".format(layer))
    

halt= True