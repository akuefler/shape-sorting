from shapesorting import *
from sandbox.util import Saver

from autoencoding.autoencoder_lib import ENCODER

import tensorflow as tf
from main import FLAGS, get_agent
import sys

from sandbox.util import Saver

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dqn_time',type=str,default='00-00-00')
args = parser.parse_args()

f = FLAGS
flags_passthrough = f._parse_flags()
agent = get_agent(sys.argv[:1] + flags_passthrough)

agent_params = sorted([(w,b) for w, b in zip(agent.wp.keys(),agent.wp.values())])
agent_biases = agent_params[0::2]
agent_weights = agent_params[1::2]
agent_params = [val[-1] for pair in zip(agent_weights, agent_biases) for val in pair]

AGENT_PARAMS = agent_params
AGENT_PARAMS_DICT = agent.wp

dqn_saver = Saver(path='{}/{}'.format(DATADIR,'dqn_weights'))

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

for k, v in f.__dict__['__flags'].iteritems():
    setattr(args,k,v)

dqn_saver.save_args(args)
dqn_saver.save_dict(0, AGENT_PARAMS_DICT, name= 'encoder')

#ENCODER.set_weights(agent_params)
#DQN_ENCODER = ENCODER