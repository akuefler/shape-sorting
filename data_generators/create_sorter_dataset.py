from shapesorting import PROJDIR

import gym

from util import ShapeSorterWrapper
from game_settings import SHAPESORT_ARGS1

from rllab.algos.trpo import TRPO
from rllab.envs.gym_env import GymEnv
from rllab.sampler.utils import rollout

from sandbox.util import Saver
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--episodes',type=int,default=10000)
parser.add_argument('--max_path_length',type=int,default=200)
parser.add_argument('--frames_per_traj',type=int,default=5)

parser.add_argument('--is_valid',type=bool,default=False)

args = parser.parse_args()

assert args.frames_per_traj < args.max_path_length

ShapeSorterWrapper.set_initials(**SHAPESORT_ARGS1)

gym.envs.register(
    id= "ShapeSorter-v0",
    #entry_point='rltools.envs.julia_sim:FollowingWrapper',
    entry_point='util:ShapeSorterWrapper',
    timestep_limit=15000,
    reward_threshold=15000,
)

env = gym.make('ShapeSorter-v0')
env = GymEnv('ShapeSorter-v0')

class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space
    def get_action(self,observation):
        return self.action_space.sample(), {}
    def reset(self):
        pass

agent= RandomPolicy(env.action_space)
T = rollout(env, agent, max_path_length=500, animated=False, flatten=False, speedup=1000)
    
saver = Saver('random_rollout_shapes', path='{}/data'.format(PROJDIR))

Et = []
Ev = []
D = {}
for E_name, E_list in zip(['train','valid'], [Et,Ev]):
    print '{} episodes ...'.format(E_name)
    for episode in range(args.episodes):
        print 'episode: {}/{}'.format(episode, args.episodes)
        observations = rollout(env, agent, max_path_length=args.max_path_length, animated=False, flatten=False)['observations']
        observations = observations[np.random.choice(observations.shape[0], args.frames_per_traj, replace=False), :]
        E_list.append(observations)
    
    X = np.concatenate(E_list,axis=0)
    D[E_name] = X
    
saver.save_args(args)
saver.save_dict(0, D)
    
halt= True