from shapesorting import *

from autoencoding.autoencoder_lib import AutoEncoder
from sandbox.util import Saver

import argparse
import tensorflow as tf
import numpy as np

#from dqn_master.restore import AGENT_PARAMS, AGENT_PARAMS_DICT

parser = argparse.ArgumentParser()

# encoding models
parser.add_argument('--similarity_time',type=str,default='16-11-06')
parser.add_argument('--autoencoder_time',type=str,default='00-00-00')
parser.add_argument('--dqnencoder_time',type=str,default='16-11-06')

# hyperparameters
parser.add_argument('--batch_size',type=int,default=20)

args = parser.parse_args()

# save to ...
encoding_saver = Saver(path='{}/{}'.format(DATADIR,'enco_simi_data'))

# load from ...
similarity_saver = Saver(time=args.similarity_time,path='{}/{}'.format(DATADIR,'simi_data'))
#autoencoder_saver = Saver(time=args.autoencoder_time,path='{}/{}'.format(DATADIR,'aue_weights'))
dqnencoder_saver = Saver(time=args.dqnencoder_time,path='{}/{}'.format(DATADIR,'dqn_weights'))

simi_data = similarity_saver.load_dictionary(0,'data')
#X1 = similarity_saver.load_value(0, "X1")
#X2 = similarity_saver.load_value(0, "X2")
#Y = similarity_saver.load_value(0, "Y")
X1 = simi_data['X1']
X2 = simi_data['X2']

N = X1.shape[0]

X1 = np.repeat(
    X1[...,None],repeats=4,axis=-1
    )
X2 = np.repeat(
    X2[...,None],repeats=4,axis=-1
    )

#autoencoder_weights = autoencoder_saver.load_dictionary(0,'encoder')
dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

#with tf.variable_scope('baseline'):
    #baseline_encoder = AutoEncoder(20)
with tf.variable_scope('reinforc'):
    reinforc_encoder = AutoEncoder(20)

with tf.Session() as sess:
    #baseline_encoder.load_weights(autoencoder_weights)
    reinforc_encoder.load_weights(dqnencoder_weights)
    #W = DQN_ENCODER.get_weights()
    #reinforc_encoder.load_weights(AGENT_PARAMS_DICT)

    for layer in reinforc_encoder.layers.keys():
        #bZ1 = []
        #bZ2 = []
        rZ1 = []
        rZ2 = []
        for i in range(0,N,args.batch_size):
            X1_batch = X1[i:i+args.batch_size]
            X2_batch = X2[i:i+args.batch_size]
            
            #bz1 = baseline_encoder.encode(X1_batch, layer=layer)
            #bz2 = baseline_encoder.encode(X2_batch, layer=layer)
            rz1 = reinforc_encoder.encode(X1_batch, layer=layer)
            rz2 = reinforc_encoder.encode(X2_batch, layer=layer)
            
            #bZ1.append(bz1)
            #bZ2.append(bz2)
            rZ1.append(rz1)
            rZ2.append(rz2)
            
        #D = {'bZ1':np.concatenate(bZ1,axis=0),
             #'bZ2':np.concatenate(bZ2,axis=0),
             #'rZ1':np.concatenate(bZ1,axis=0),
             #'rZ2':np.concatenate(bZ2,axis=0)}
        RZ1 = np.concatenate(rZ1,axis=0)
        RZ2 = np.concatenate(rZ2,axis=0)
        D = {'rZ1':RZ1,
             'rZ2':RZ2}
        encoding_saver.save_dict(0, D, name="{}_encodings".format(layer))
    encoding_saver.save_dict(0, simi_data, name='simi_data')
    
halt= True