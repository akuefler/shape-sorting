class AgentConfig(object):
  scale = 10000
  #scale = 1
  display = False

  max_tsteps= 500
  #max_step = 5000 * scale
  max_step = 10000 * scale  
  memory_size = 100 * scale

  batch_size = 32
  random_start = 30
  cnn_format = 'NCHW'
  #discount = 0.99
  discount = 0.95
  target_q_update_step = 1 * scale
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  #history_length = 1
  history_length = 1
  
  train_frequency = 4
  learn_start = 5. * scale

  #learn_start = 10 * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False
  extra_hidden = False

  _test_step = 5 * scale
  #_save_step = 250 * scale
  _save_step = 125 * scale
  

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1000.
  min_reward = -1000.
  

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  folder_name= 'default'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
