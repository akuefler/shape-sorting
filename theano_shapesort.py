import gym
import calendar

from rllab.sampler.utils import rollout

from util import ShapeSorterWrapper
from game_settings import SHAPESORT_ARGS

from rllab.algos.trpo import TRPO
from rllab.envs.gym_env import GymEnv
#from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy

from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab import RLLabRunner

import argparse
import lasagne.nonlinearities as NL

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

# shapesorting params
parser.add_argument('--shapesort_args',type=int,default=4)
parser.add_argument('--show_once',type=bool,default=False)

# hyperparams
parser.add_argument('--max_path_length',type=int,default=500)
parser.add_argument('--batch_size',type=int,default=20)
parser.add_argument('--n_iter',type=int,default=500)
parser.add_argument('--nonlinearity',type=str,default='lrelu')

parser.add_argument('--hspec',type=int,nargs='+',default=[20, 10, 10])
parser.add_argument('--filt_d',type=int,nargs='+',default=[16,6])
parser.add_argument('--filt_hw',type=int,nargs='+',default=[4,4])
parser.add_argument('--filt_s',type=int,nargs='+',default=[2,1])
parser.add_argument('--filt_p',type=int,nargs='+',default=[0,0])

parser.add_argument('--discount',type=float,default=0.99)

parser.add_argument('--args_data',type=str)

args = parser.parse_args()

args.exp_name += calendar.datetime.datetime.now().strftime("-%y-%m-%d-%I-%M%p")

ShapeSorterWrapper.set_initials(**SHAPESORT_ARGS[args.shapesort_args])

import lasagne

nonlinearities = {'relu':NL.rectify,
                  'lrelu':NL.leaky_rectify,
                  'tanh':NL.tanh,
                  'sigmoid':NL.sigmoid}

gym.envs.register(
    id= "ShapeSorter-v0",
    #entry_point='rltools.envs.julia_sim:FollowingWrapper',
    entry_point='util:ShapeSorterWrapper',
    timestep_limit=15000,
    reward_threshold=15000,
)

#env = gym.make('ShapeSorter-v0')

env = GymEnv('ShapeSorter-v0')
                     
policy = CategoricalConvPolicy('policy', env.spec,
                               conv_filters=args.filt_d,
                               conv_filter_sizes=args.filt_hw,
                               conv_strides=args.filt_s,
                               conv_pads=args.filt_p,
                               hidden_nonlinearity=nonlinearities[args.nonlinearity],
                               hidden_sizes=args.hspec)

if args.baseline_type == 'mlp':
    baseline = GaussianConvBaseline(env.spec,regressor_args= dict(
    conv_filters=args.filt_d,conv_filter_sizes=args.filt_hw,conv_strides=args.filt_s,
    conv_pads=args.filt_p,hidden_sizes=args.hspec,use_trust_region=False,
    optimizer=FirstOrderOptimizer(max_epochs=10,batch_size=10)))

max_path_length = args.max_path_length
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    #batch_size=4000,
    batch_size= args.batch_size * max_path_length,
    max_path_length=max_path_length,
    n_itr=args.n_iter,
    discount=args.discount,
    step_size=0.01,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

if args.show_once:
    rollout(env, policy, animated=True)

runner = RLLabRunner(algo, args)
runner.train()
