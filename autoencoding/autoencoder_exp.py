import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--exp_name',type=str,default='shape_encoder')
parser.add_argument('--data_name',type=str,default='random_rollout_shapes-00004')

parser.add_argument('--normalize',type=bool,default=True)
parser.add_argument('--share_weights',type=bool,default=True)
parser.add_argument('--n_channels',type=int,default=4)

parser.add_argument('--train',type=bool,default=False)

# training
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=20)
parser.add_argument('--cmx',type=float,default=1.0) # regularization weight for complexity loss

args = parser.parse_args()

assert args.batch_size == 20
    
modelsaver = Saver(args.exp_name, path='data', overwrite= False)
datasaver = Saver(args.data_name, path='data', overwrite=True)
XT = datasaver.load_value(0,'train')
XV = datasaver.load_value(0,'valid')
if args.n_channels > 1:
    XT = np.repeat(XT[...,None], args.n_channels, axis=-1)
    XV = np.repeat(XV[...,None], args.n_channels, axis=-1)

model = AutoEncoder(args.batch_size)

if args.train:
    pt = np.random.permutation(XT.shape[0])
    pv = np.random.permutation(XV.shape[0])
    X_t = XT[pt]
    X_v = XV[pv]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        model.train(X_t, X_v, saver=modelsaver,epochs=args.epochs)
    #train(model, X_t, X_v, batch_size=args.batch_size, epochs=args.epochs, saver= modelsaver)
    