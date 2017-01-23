import random
import tensorflow as tf

import sys
sys.path.insert(1,"/home/alex/Desktop/stanford_dev/ShapeSorting/shapesorting-master")

from dqn.agent import Agent
from dqn.environment import GymEnvironment, SimpleGymEnvironment, ShapeSorterEnvironment
from config import get_config
from game import ShapeSorter

flags = tf.app.flags

DATADIR="/home/alex/stanford_dev/thesis/win/shapesorting/data/"

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
dueling= True
flags.DEFINE_boolean('dueling', dueling, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# Environment
#flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
#flags.DEFINE_string('env_name', 'CartPole-v0', 'The name of gym environment to use')
flags.DEFINE_string('env_name', 'shapesort', 'The name of gym environment to use')
flags.DEFINE_string('env_type', 'shapesort', 'environment type?')
game_settings = 2
flags.DEFINE_string('game_settings', game_settings, 'game settings')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')

name = "october28_dueling_settings1"
#name = 'december08_duel{dueling}_settings{settings}_unitElu'.format(dueling=int(dueling),settings=game_settings)

flags.DEFINE_string('folder_name', name, 'The name of the folder to save to.')
#flags.DEFINE_string('folder_name', 'july17', 'The name of the folder to save to.')

#flags.DEFINE_string('folder_name', 'breakout_test', 'The name of the folder to save to.')

# Etc

flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
#flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')

flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print " [*] GPU : %.4f" % fraction
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))
  from game_settings import SHAPESORT_ARGS
  settings = SHAPESORT_ARGS[FLAGS.game_settings]

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    if config.env_type =='shapesort':
      env = ShapeSorterEnvironment(config, settings)
    else:
      env = GymEnvironment(config)

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    agent = Agent(config, env, sess)

    if settings["experiment"] in ["training"]:
      agent.train()
    elif settings["experiment"] in ["preference","one_block"]:
      agent.play(settings["experiment"], data_dir= DATADIR)
      
def get_agent(_, load_weights):
  with tf.Session() as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    if config.env_type =='shapesort':
      from game_settings import SHAPESORT_ARGS
      env = ShapeSorterEnvironment(config,
                                   SHAPESORT_ARGS[FLAGS.__dict__['__flags']['game_settings']])
    else:
      env = GymEnvironment(config)

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    agent = Agent(config, env, sess, load_weights= load_weights)
    return agent
  

if __name__ == '__main__':
  #tf.app.run()
  tf.app.run()
#else:
  #AGENT = tf.app.run(main=get_agent)
