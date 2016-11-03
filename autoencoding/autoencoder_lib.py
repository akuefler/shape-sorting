import numpy as np
import tensorflow as tf
from game import ShapeSorter
from game_settings import SHAPESORT_ARGS
import keras as ks
import keras.backend as K
import keras.layers as KL

from dqn_master.dqn.ops import linear, conv2d

import random

from game_settings import DISCRETE_ACT_MAP4
from rllab.sampler.utils import rollout

assert K.backend() == 'tensorflow'

from sandbox.util import Saver

env = ShapeSorter(**SHAPESORT_ARGS[1])

input_shape= (84,84,4)  # TENSORFLOW INPUT SHAPE

#bias = K.variable(np.zeros((output_shape[-1],)))
#model.add(KL.Lambda(lambda x : x + bias, output_shape= output_shape[1:]))

class BiasLayer(KL.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = K.variable(np.zeros((self.output_dim,)))
        self.trainable_weights = [self.b]

    def call(self, x, mask=None):
        if len(x.get_shape()) > 2:
            b = tf.reshape(self.b, x.get_shape()[1:])
        else:
            b = self.b
        return x + b

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

def create_conv(tensor,conv_hw,conv_s,conv_d,activation='relu',batch_size=20,model_name=None):
    scope = tf.get_variable_scope()    
    
    for i, (hw, stride, n_filters) in enumerate(zip(conv_hw,conv_s,conv_d)):
        name = '{}/conv{}'.format(scope.name,i)
        tensor = KL.Conv2D(n_filters, hw, hw, subsample=(stride,stride), activation=activation, name= name)(tensor)
            
    tensor = KL.Flatten()(tensor)
            
    return tensor

def _create_sequential_encoder(conv_hw,conv_s,conv_d, hspec,
                               activation='relu',batch_size=20,model_name=None,
                               layer_names=['conv0','conv1','conv2','fc0','fc1']):
    """
    create a dummy model which we don't train so we can easily count output layer sizes
    of the convolutions. (Eases specification of the deconv net).
    """
    model = ks.models.Sequential()
    for i, (hw, stride, n_filters) in enumerate(zip(conv_hw,conv_s,conv_d)):
        if i == 0:
            model.add(KL.Conv2D(n_filters, hw, hw, subsample=(stride,stride),input_shape= (input_shape), name= layer_names.pop(0)))
        else:
            model.add(KL.Conv2D(n_filters, hw, hw, subsample=(stride,stride),
                            activation=activation, name=layer_names.pop(0)))        
    model.add(KL.Flatten())    
        
    for i, h in enumerate(hspec):
        model.add(KL.Dense(h,activation='relu',name=layer_names.pop(0)))
                            
    return model


def create_deconv(tensor,output_shapes,conv_hw,conv_s,conv_d,activation='relu',model_name=None):
    #model.add(KL.Reshape((7,7,64))) # fix this ...
    scope = tf.get_variable_scope()
    tensor = KL.Reshape((7,7,64))(tensor)
    
    for i, (hw, stride, n_filters) in enumerate(zip(conv_hw,conv_s,conv_d)):
        name = '{}/deconv{}'.format(scope.name,i)        
        output_shape = output_shapes[i]

        tensor = KL.Deconv2D(n_filters, hw, hw, output_shape=output_shape, subsample=(stride,stride),
                             activation=None,name=name,bias=False)(tensor)
        
        bias = tf.Variable(initial_value= np.zeros((output_shape[-1],)),dtype=tf.float32)
        bias = K.variable(np.zeros((output_shape[-1],)))
        tensor = KL.Lambda(lambda x : x + bias, output_shape= output_shape[1:])(tensor)
        #model.add(BiasLayer(np.prod(output_shape[1:])))
        
        if i != len(output_shape):
            tensor = KL.Activation(activation)(tensor)
        
    return tensor

def create_mlp(tensor,hspec,activation='relu'):
    scope = tf.get_variable_scope()
    
    for i, h in enumerate(hspec):
        if i == len(hspec) - 1:
            activation = None
        name = '{}/fc{}'.format(scope.name,i)
        tensor = KL.Dense(h,activation=None,name=name,bias=False)(tensor)
        tensor = BiasLayer(h)(tensor)
        tensor = KL.Activation(activation)(tensor)

    return tensor

class AutoEncoder(object):
    def __init__(self, batch_size):
        
        self.batch_size = batch_size
        
        self.layers = {}
        self.encoder_weights = {}
        self.decoder_weights = {}
        activation_fn = tf.nn.relu
        self.x = x = tf.placeholder(tf.float32, shape=(batch_size,84,84,4))
        
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        
        with tf.variable_scope('autoencoder') as scope:
            #ENCODER
            self.l1, self.encoder_weights['l1_w'], self.encoder_weights['l1_b'] = conv2d(x,
                                                             32, [8, 8], [4, 4], initializer, activation_fn, 'NHWC', name='l1')
            self.l2, self.encoder_weights['l2_w'], self.encoder_weights['l2_b'] = conv2d(self.l1,
                                                             64, [4, 4], [2, 2], initializer, activation_fn, 'NHWC', name='l2')
            self.l3, self.encoder_weights['l3_w'], self.encoder_weights['l3_b'] = conv2d(self.l2,
                                                             64, [3, 3], [1, 1], initializer, activation_fn, 'NHWC', name='l3')
        
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
            
            self.l4, self.encoder_weights['l4_w'], self.encoder_weights['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
            self.q, self.encoder_weights['q_w'], self.encoder_weights['q_b'] = linear(self.l4, 7, name='q')
            
            self.layers['l1'] = self.l1
            self.layers['l2'] = self.l2
            self.layers['l3'] = self.l3
            self.layers['l4'] = self.l4
            self.layers['q'] = self.q
            
            #DECODER
            output_shapes = [(20, 9, 9, 64), (20, 20, 20, 32), (20, 84, 84, 4)]
            
            self.decoder_weights['q_w_t'] = q_w_t = tf.transpose(self.encoder_weights['q_w'])
            self.decoder_weights['l4_w_t'] = l4_w_t = tf.transpose(self.encoder_weights['l4_w'])
            self.decoder_weights['l3_w_t'] = l3_w_t = tf.transpose(self.encoder_weights['l3_w'],[0,1,2,3])
            self.decoder_weights['l2_w_t'] = l2_w_t = tf.transpose(self.encoder_weights['l2_w'],[0,1,2,3])
            self.decoder_weights['1l_w_t'] = l1_w_t = tf.transpose(self.encoder_weights['l1_w'],[0,1,2,3])
            
            self.decoder_weights['q_b_t'] = q_b_t = tf.Variable(np.zeros(q_w_t.get_shape()[-1].value,).astype('float32'))
            self.decoder_weights['l4_b_t'] = l4_b_t = tf.Variable(np.zeros(64,).astype('float32')) # derp.
            self.decoder_weights['l3_b_t'] = l3_b_t = tf.Variable(np.zeros(l3_w_t.get_shape()[-2].value,).astype('float32'))
            self.decoder_weights['l2_b_t'] = l2_b_t = tf.Variable(np.zeros(l2_w_t.get_shape()[-2].value,).astype('float32'))
            self.decoder_weights['l1_b_t'] = l1_b_t = tf.Variable(np.zeros(l1_w_t.get_shape()[-2].value,).astype('float32'))
            
            scope.reuse_variables()
            self.q_T = activation_fn(tf.nn.xw_plus_b(self.q, tf.transpose(self.encoder_weights['q_w']), q_b_t, name='q_T'))

            self.l4_T = tf.matmul(self.q_T, tf.transpose(self.encoder_weights['l4_w']))
            self.l4_T_unflat = activation_fn( tf.reshape(self.l4_T, self.l3.get_shape()) + l4_b_t)
            
            self.l3_T = activation_fn(tf.nn.conv2d_transpose(self.l4_T_unflat, l3_w_t, output_shapes[0], 
                                              [1,1,1,1],padding="VALID",name='l3_T') + l3_b_t)
            self.l2_T = activation_fn(tf.nn.conv2d_transpose(self.l3_T, l2_w_t, output_shapes[1], 
                                                          [1,2,2,1],padding="VALID",name='l3_T') + l2_b_t)
            self.l1_T = tf.nn.conv2d_transpose(self.l2_T, l1_w_t, output_shapes[2], 
                                                          [1,4,4,1],padding="VALID",name='l1_T')+ l1_b_t
            
            self.x_hat = self.l1_T
            self.cost = tf.reduce_mean(
                tf.nn.l2_loss(self.x_hat - self.x)
                )
            self.opt = tf.train.AdamOptimizer().minimize(self.cost)
            
    def predict(self, X):
        sess = tf.get_default_session()
        X_hat = sess.run(self.x_hat,{self.x:X})
        return X_hat
    
    def encode(self, X, layer= 'l4'):
        sess = tf.get_default_session()
        z = sess.run(self.layers[layer],{self.x:X})
        return z
    
    def load_weights(self, weights):
        convert = {'l1_w':'l1_W:0',
                   'l2_w':'l2_W:0',
                   'l3_w':'l3_W:0',
                   'l4_w':'l4_W:0',
                   'q_w':'q_W:0',
                   'l1_b':'l1_b:0',
                   'l2_b':'l2_b:0',
                   'l3_b':'l3_b:0',
                   'l4_b':'l4_b:0',
                   'q_b':'q_b:0',          
                   }
        ops = []
        for k, v in weights.iteritems():
            ops.append(
                self.encoder_weights[k].assign(v)
                )
        sess = tf.get_default_session()
        sess.run(ops)
    
    def train(self, X_t, X_v, epochs=1000, saver= None):
        sess = tf.get_default_session()
        N = X_t.shape[0]
        N_v = X_v.shape[0]
        
        for epoch in range(epochs):
            p = np.random.permutation(X_t.shape[0])
            X_t = X_t[p]
            losses_t = []
            losses_v = []
            for i in range(0,N,self.batch_size):
                X_t_batch = X_t[i:self.batch_size+i]
                loss, _ = sess.run([self.cost,self.opt],{self.x: X_t_batch})
                losses_t.append(loss)
            for i in range(0,N_v,self.batch_size):
                X_v_batch = X_v[i:self.batch_size+i]
                loss = sess.run(self.cost,{self.x: X_v_batch})
                losses_v.append(loss) 
                
            loss_t = np.mean(losses_t)
            loss_v = np.mean(losses_v)
            
            print 'Epoch: {}/{}, Training loss: {}, validation loss: {}'.format(epoch,epochs,loss_t,loss_v)
            
            if saver is not None:
                #saver.save_models(epoch, [self])
                saver.save_dict(epoch, {'loss_t': loss_t, 'loss_v': loss_v})
                encoder_d = {key:tensor.eval(session=sess) for key, tensor in self.encoder_weights.iteritems()}
                decoder_d = {key:tensor.eval(session=sess) for key, tensor in self.decoder_weights.iteritems()}
                saver.save_dict(epoch, decoder_d, name= 'decoder')
                saver.save_dict(epoch, encoder_d, name= 'encoder')
                


def autoencoder(hspec,batch_size,conv_hw,conv_s,conv_d,share_weights=True):
    
    # encoder (1687200 params)
    # DQN : 1687719
    with tf.variable_scope('encoder'):
        enc_input = KL.Input(batch_shape= (batch_size,) + input_shape)
        enc_conv = create_conv(enc_input, conv_hw, conv_s, conv_d, activation='relu', batch_size= batch_size)
        enc_mlp = create_mlp(enc_conv, hspec, activation='relu')
    
    _fnc_encoder = ks.models.Model(enc_input, enc_mlp)
    _seq_encoder = _create_sequential_encoder(conv_hw, conv_s, conv_d, hspec) # encoder that can be inverted for convenience
    assert _fnc_encoder.count_params() == _seq_encoder.count_params()
    
    tspec = []
    ispec = [l.input_shape[-1] for l in _seq_encoder.layers[::-1] if type(l)==KL.Dense]
    output_shapes = [(batch_size,)+l.input_shape[1:] for l in _seq_encoder.layers[::-1] if type(l)== KL.convolutional.Convolution2D]
    
    with tf.variable_scope('decoder'):
        dec_mlp = create_mlp(enc_mlp, ispec, activation='relu')
        dec_conv = create_deconv(dec_mlp, output_shapes, conv_hw[::-1], conv_s[::-1], conv_d[1::-1] + [4],
                                 activation='relu')
    
    model = ks.models.Model(enc_input, dec_conv, name= 'autoencoder')
    
    if share_weights:
        model = share_decoder_weights(model)
        
    def ae_mse_loss(y_true, y_pred):
        
        y_t = tf.reshape(y_true, (batch_size,-1))
        y_p = tf.reshape(y_pred, (batch_size,-1))
        if share_weights:
            encoder_layers = filter(lambda x : x.name.split('/')[0] == 'encoder', model.layers)
            decoder_layers = filter(lambda x : x.name.split('/')[0] == 'decoder', model.layers)
            
            for el, dl in zip(encoder_layers,decoder_layers[::-1]):
                dl.trainable= False
                w = el.get_weights()[0]
                dl.set_weights([np.swapaxes(w, -1, -2)])            
            
        return ks.objectives.mse(y_t, y_p)

    print model.count_params()
    model.compile(optimizer='adam', loss=ae_mse_loss)
    return model, _seq_encoder

def share_decoder_weights(model):
    encoder_layers = filter(lambda x : x.name.split('/')[0] == 'encoder', model.layers)
    decoder_layers = filter(lambda x : x.name.split('/')[0] == 'decoder', model.layers)
    
    for el, dl in zip(encoder_layers,decoder_layers[::-1]):
        dl.trainable= False
        
        w = el.get_weights()[0]
        dl.set_weights([np.swapaxes(w, -1, -2)])
        
    return model

def train(model, X_t, X_v, batch_size= 20, epochs= 100, saver= None):
    N = X_t.shape[0]
    
    Xp_t = X_t.copy()
    Xp_v = X_v.copy()
    
    sess = tf.Session()
    
    for epoch in xrange(epochs):
        p = np.random.permutation(N)
        Xp_t = Xp_t[p]
        Xp_v = Xp_v[p]
        
        for i in range(0,N,batch_size):
            Xp_t_batch = Xp_t[i:i+batch_size]
            Xp_v_batch = Xp_v[i:i+batch_size]
            
            _ = model.train_on_batch(Xp_t_batch, Xp_t_batch)
            
        loss_t, loss_v = sess.run(
            [tf.reduce_mean(model.loss(Xp_t_batch, model.predict_on_batch(Xp_t_batch))),
             tf.reduce_mean(model.loss(Xp_v_batch, model.predict_on_batch(Xp_v_batch)))]
            )
        
        print "Epoch {}, average training loss: {}, validation loss: {}".format(epoch,loss_t,loss_v)
            
        if saver is not None:
            saver.save_models(epoch, [model])
            saver.save_dict(epoch, {'loss_t': loss_t, 'loss_v': loss_v})
            

#DQNCODER, _ = autoencoder([512,len(DISCRETE_ACT_MAP4)], batch_size=20, conv_hw=[8,4,3], conv_s=[4,2,1], conv_d=[32,64,64],
                    #share_weights= False)

#DQNCODER_SHARE, _ = autoencoder([512,len(DISCRETE_ACT_MAP4)], batch_size=20, conv_hw=[8,4,3], conv_s=[4,2,1], conv_d=[32,64,64],
                    #share_weights= False)

ENCODER = _create_sequential_encoder([8,4,3], [4,2,1], [32,64,64], [512, len(DISCRETE_ACT_MAP4)],
                                     activation='relu', batch_size=20, model_name='encoder',
                                     layer_names=['l1','l2','l3','l4','q'])

