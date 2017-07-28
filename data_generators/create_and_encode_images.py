from config import *

from autoencoding.autoencoder_lib import AutoEncoder
from util import Saver

from game import create_renderList, process_observation
from initializers import grid_initializer
from game_settings import SHAPESORT_ARGS, INITIALIZER_MAP

import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pygame as pg

parser = argparse.ArgumentParser()

# encoding models
parser.add_argument('--dqnencoder_time',type=str,default='16-11-11-07-06PM') # trained network
#parser.add_argument('--dqnencoder_time',type=str,default='17-01-29-14-12-28-335751') # UNTRAINED network

# hyperparameters
parser.add_argument('--batch_size',type=int,default=10)
parser.add_argument('--N',type=int,default=10000)
parser.add_argument('--settings',type=int,default=2)

parser.add_argument('--include_holes',type=int,default=1)
parser.add_argument('--vary_block_center',type=int,default=1)
parser.add_argument('--grab',type=int,default=0)

parser.add_argument('--enumerate_scenes',type=int,default=1)

######
#main
######
args = parser.parse_args()
shapesort_args = SHAPESORT_ARGS[args.settings]

# save to ...
encoding_saver = Saver(path='{}/{}'.format(DATADIR,'scene_and_enco_data'))
# load from ...
dqnencoder_saver = Saver(time=args.dqnencoder_time,path='{}/{}'.format(DATADIR,'dqn_weights'))
dqnencoder_weights = dqnencoder_saver.load_dictionary(0,'encoder')

screen = pg.display.set_mode((shapesort_args['screen_HW'],shapesort_args['screen_HW']))

if args.enumerate_scenes:
    Ds = grid_initializer(0, shapesort_args['shapes'], shapesort_args['sizes'], shapesort_args['rot_size'], permute=100)
    scope = len(Ds)
else:
    scope = args.N

X, H, B = [], [], []
for i in xrange(scope):
    screen.fill((255,255,255))

    if args.enumerate_scenes:
        D = Ds[i]
    else:
        D = INITIALIZER_MAP[shapesort_args['experiment']](shapesort_args['n_blocks'],
                                                          shapesort_args['shapes'],
                                                          shapesort_args['sizes'],
                                                          shapesort_args['rot_size'],
                                                          shapesort_args['random_holes'])
    hList, bList = create_renderList(D, shapesort_args['screen_HW'], shapesort_args['screen_HW'])
    if args.include_holes:
        rList = hList + bList
    else:
        rList = bList

    for item in rList:
        if item.typ == "block" and not args.vary_block_center:
            item.center = (shapesort_args['screen_HW']/2, shapesort_args['screen_HW']/2)
        item.render(screen, angle=5.0) # Draw all items
        if item.typ == "block" and args.grab > 0:
            # blue, green
            col = [(0,0,255),(0,255,0)][args.grab - 1]
            pg.draw.circle(screen, col, item.center, shapesort_args['cursor_size'])

    #pg.display.flip()
    hole_dict = {'shape':[shapesort_args['shapes'].index(type(h)) for h in hList],
                 'angle':[h.angle for h in hList],
                 'center':[h.center for h in hList]}
    block_dict = {'shape':[shapesort_args['shapes'].index(type(b)) for b in bList],
                  'angle':[b.angle for b in bList],
                  'center':[b.center for b in bList]}
    x = process_observation(screen, shapesort_args['screen_HW'], shapesort_args['screen_rHW'])
    #plt.imshow(x)
    #plt.show()

    X.append(x[None,...])
    H.append(hole_dict)
    B.append(block_dict)

X = np.concatenate(X,axis=0)
X = np.repeat(
    X[...,None],repeats=4,axis=-1
    )

BD = dict(zip(map(lambda x: "{:0{}}".format(x,len(str(scope))),range(len(B))),B))
HD = dict(zip(map(lambda x: "{:0{}}".format(x,len(str(scope))),range(len(H))),H))

with tf.variable_scope('reinforc'):
    reinforc_encoder = AutoEncoder(args.batch_size, supervised=False)

#ZD = {}
D = {}
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    reinforc_encoder.load_weights(dqnencoder_weights)

    keys = ['l1_flat','l2_flat','l3_flat','adv_hid','value_hid']
    for j, layer in enumerate(keys):
        Z = []
        for i in range(0,scope,args.batch_size):
            if i % 1000 == 0:
                print("Layer: {} of {} == Example: {} of {}".format(j, len(keys), i, scope))
            x_batch = X[i:i+args.batch_size]
            z = reinforc_encoder.encode(x_batch, layer=layer)
            Z.append(z)
        Z = np.concatenate(Z,axis=0)
        D.update({"Z_{}".format(layer): Z})
    D.update({'X':X})
    encoding_saver.save_dict(0, D, name='data')
    encoding_saver.save_recursive_dict(0, BD, name='blocks')
    encoding_saver.save_recursive_dict(0, HD, name='holes')
    encoding_saver.save_args(args)

halt= True
