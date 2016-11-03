import gym
from util import ShapeSorterWrapper
from game_settings import SHAPESORT_ARGS1

from rllab.algos.trpo import TRPO
from rllab.envs.gym_env import GymEnv
#from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy

from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab import RLLabRunner

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load_itr_path',type=str,default=None)

# joblib params
parser.add_argument('--baseline_type',type=str,default='mlp')
parser.add_argument('--exp_name',type=str,default='trpo_shapesorter')
parser.add_argument('--tabular_log_file',type=str,default= 'tab.txt')
parser.add_argument('--text_log_file',type=str,default= 'tex.txt')
parser.add_argument('--params_log_file',type=str,default= 'args.txt')
parser.add_argument('--snapshot_mode',type=str,default='all')
parser.add_argument('--log_tabular_only',type=bool,default=False)
parser.add_argument('--log_dir',type=str)

parser.add_argument('--args_data',type=str)

args = parser.parse_args()


ShapeSorterWrapper.set_initials(**SHAPESORT_ARGS1)

gym.envs.register(
    id= "ShapeSorter-v0",
    #entry_point='rltools.envs.julia_sim:FollowingWrapper',
    entry_point='util:ShapeSorterWrapper',
    timestep_limit=15000,
    reward_threshold=15000,
)

#env = gym.make('ShapeSorter-v0')

env = GymEnv('ShapeSorter-v0')

#policy = CategoricalConvPolicy('policy', env.spec, [16, 32], [8,4], 
                     #[4,2], [0,0], hidden_sizes=[256])
                     
policy = CategoricalConvPolicy('policy', env.spec,
                               conv_filters = [32,64,64],
                               conv_filter_sizes=[8,4,3],
                               conv_strides=[4,2,1],
                               conv_pads=[0,0,0],
                               hidden_sizes=[512])

if args.baseline_type == 'mlp':
    baseline = GaussianConvBaseline(env.spec,regressor_args= dict(
        conv_filters=[16,32],conv_filter_sizes=[8,4],conv_strides=[4,2],
        conv_pads=[0,0],hidden_sizes=[256],use_trust_region=False,
        optimizer=FirstOrderOptimizer(max_epochs=10,batch_size=1)))

max_path_length = 500
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    #batch_size=4000,
    batch_size=20 * max_path_length,
    max_path_length=max_path_length,
    n_itr=500,
    discount=0.99,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

runner = RLLabRunner(algo, args)
runner.train()

halt= True

import theano