from shapesorting import *

import argparse

import numpy as np
import tensorflow as tf

from shapesorting.autoencoding.autoencoder_lib import *
from sandbox.util import Saver

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument('--load_dqn',type=bool,default=False)

#parser.add_argument('--exp_name',type=str,default='end')
parser.add_argument('--share_weights',type=bool,default=True)
parser.add_argument('--n_channels',type=int,default=4)

parser.add_argument('--train',type=bool,default=False)

# training
parser.add_argument('--epochs',type=int,default=2000)
parser.add_argument('--batch_size',type=int,default=20)

#parser.add_argument('--simi_data_time',type=str,default="16-11-10-01-13AM")
parser.add_argument('--simi_data_time',type=str,default="16-11-10-08-58PM")

args = parser.parse_args()

assert args.batch_size == 20
    
simidata_saver = Saver(time=args.simi_data_time,path='{}/{}'.format(DATADIR,'simi_data'))
modelsaver = Saver(path='{}/{}'.format(DATADIR,'classifier'))

D = simidata_saver.load_dictionary(0,'data')
#XV = simidata_saver.load_value(0,'valid')
if args.n_channels > 1:
    X1, X2, Y = D['X1'], D['X2'], D['Y']
    
    ## fake data
    #X1 = np.concatenate([np.zeros((600,84,84)),
                        #np.zeros((600,84,84))], axis= 0)
    #X2 = np.concatenate([np.ones((600,84,84)),
                        #np.zeros((600,84,84))], axis= 0)
    #Y = np.concatenate([np.zeros(600,),np.ones(600,)])
    
    ## real data
    p = np.random.choice(X1.shape[0],20 * 1000,replace=False)
    X1 = X1[p]
    X2 = X2[p]
    Y = Y[p]
    
    X1 = np.expand_dims(np.repeat(X1[...,None], args.n_channels, axis=-1),1)
    X2 = np.expand_dims(np.repeat(X2[...,None], args.n_channels, axis=-1),1)
    
    X = np.concatenate([X1,X2],axis=1)
        
    XT = X[:int(0.7 * len(Y))]
    YT = Y[:int(0.7 * len(Y))]
    
    XV = X[int(0.7 * len(Y)):]
    YV = Y[int(0.7 * len(Y)):]

model = AutoEncoder(args.batch_size, dueling=True, supervised=True, top_layer='l3_flat')
#model = SplitNet(args.batch_size)

if args.train:
    
    with tf.Session() as sess:
        #import matplotlib.pyplot as plt
        #i = 0
        #import matplotlib.pyplot as plt
        #i = 0
        #while True:
            #f, (ax1) = plt.subplots(1)
            #A = XT[i,0,:,:,1].astype('float32')
            #B = XT[i,1,:,:,1].astype('float32')
            #ax1.imshow(np.column_stack((A,B)))
            #plt.show()
            #print YT[i]
            
            #i += 1       

        sess.run(tf.initialize_all_variables())
        if args.load_dqn:
            dqnencoder_saver = Saver(time='16-11-06',path='{}/{}'.format(DATADIR,'dqn_weights'))
            dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')
            model.load_weights(dqnencoder_weights)        

        train(model,XT, YT, XV, YV, saver=modelsaver,epochs=args.epochs)
        
        pt, lt = model.predict(XT)
        pv, lv = model.predict(XV)
        
        #import matplotlib.pyplot as plt
        #i = 0
        #while True:
            #f, (ax1) = plt.subplots(1)
            #A = XT[i,0,:,:,1].astype('float32')
            #B = XT[i,1,:,:,1].astype('float32')
            #ax1.imshow(np.column_stack((A,B)))
            #plt.show()
            #print YT[i]
            
            #i += 1
        
        print accuracy_score(YT, pt)
        print accuracy_score(YV, pv)
        
        halt= True