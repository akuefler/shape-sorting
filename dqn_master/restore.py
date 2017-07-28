from shapesorting import *

import tensorflow as tf
from main import FLAGS, get_agent
import sys

from util import Saver
DATADIR="/home/alex/stanford_dev/thesis/win/shapesorting/data/"

import argparse

import numpy as np

parser = argparse.ArgumentParser()
#parser.add_argument('--dqn_time',type=str,default='00-00-00')
parser.add_argument('--load_weights',type=bool,default=False)
parser.add_argument('--seed',type=int,default=456)
args = parser.parse_args()

np.random.seed(args.seed)

f = FLAGS
flags_passthrough = f._parse_flags()
agent = get_agent(sys.argv[:1] + flags_passthrough, load_weights= args.load_weights)
#agent = get_agent(None)

agent_params = sorted([(w,b) for w, b in zip(agent.wp.keys(),agent.wp.values())])
agent_biases = agent_params[0::2]
agent_weights = agent_params[1::2]
agent_params = [val[-1] for pair in zip(agent_weights, agent_biases) for val in pair]

AGENT_PARAMS = agent_params
AGENT_PARAMS_DICT = agent.wp

dqn_saver = Saver(path='{}/{}'.format(DATADIR,'dqn_weights'))

for k, v in f.__dict__['__flags'].iteritems():
    setattr(args,k,v)

dqn_saver.save_args(args)
dqn_saver.save_dict(0, AGENT_PARAMS_DICT, name= 'encoder')

#ENCODER.set_weights(agent_params)
#DQN_ENCODER = ENCODER
