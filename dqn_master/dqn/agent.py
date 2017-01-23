import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, conv2d
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

from collections import Counter

from util import Saver

class Agent(BaseModel):
  def __init__(self, config, environment, sess, load_weights= True):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.load_weights = load_weights
    self.weight_dir = 'weights'

    self.env = environment
    self.env_type = config.env_type
    
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    if self.env_type == 'shapesort':
      screen = self.env.reset()
    else:
      screen, reward, action, terminal = self.env.new_random_game()      

    for _ in range(self.history_length):
      self.history.add(screen)

    ep_tsteps= 0
    for step in tqdm(range(self.max_step), ncols=70, initial=start_step):
      self.step = step + start_step
#    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict(self.history.get())
      # 2. act
      if self.env_type == 'shapesort':
        screen, reward, terminal, _ = self.env.step(action)       
      else:
        screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe
      self.observe(screen, reward, action, terminal)
      
      if self.display:
        self.env.render()
        
      if ep_tsteps >= self.max_tsteps:
        terminal= True

      if terminal:
        if self.env_type == 'shapesort':
          screen = self.env.reset()          
          
        else:
          screen, reward, action, terminal = self.env.new_random_game()

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
        ep_tsteps = 0
      else:
        ep_reward += reward
        ep_tsteps += 1

      actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        
        #Saving interval.
        if self.step % self.save_step == self.save_step - 1:
          self.step_assign_op.eval({self.step_input: self.step + 1})
          self.save_model(self.step + 1)

          #max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)        

        #Testing interval.
        if self.step % self.test_step == self.test_step - 1: #Do every "test_step" iterations.
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

          #if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            #self.step_assign_op.eval({self.step_input: self.step + 1})
            #self.save_model(self.step + 1)

            #max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                #'training.gradient_mags': self.gradient_mag_op.eval({self.})
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

  def predict(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t]})[0]

    return action

  def observe(self, screen, reward, action, terminal):
    updated_dqn= False; updated_target= False
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)

    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch()
        updated_dqn= True

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()
        updated_target
        
    return updated_dqn, updated_target

  def q_learning_mini_batch(self):
    if self.memory.count <= self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    t = time.time()
    if self.double_q:
      # Double Q-learning
      pred_action = self.q_action.eval({self.s_t: s_t_plus_1})

      q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
        self.target_s_t: s_t_plus_1,
        self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]
      })
      target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
    else:
      q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})

      terminal = np.array(terminal) + 0.
      max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
      target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

    _, q_t, loss, q_summary_str, gm_summary_str = \
      self.sess.run([self.optim, self.q, self.loss, self.q_summary, self.gm_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
    })

    self.writer.add_summary(q_summary_str, self.step)
    self.writer.add_summary(gm_summary_str,self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def build_dqn(self):
    self.w = {}
    self.t_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')

      #output_dims = [16,32,32]
      output_dims = [32,64,64]
      hw = [[8,8],[4,4],[3,3]]
      strides = [[4,4],[2,2],[1,1]]
      hidsize = [512]

      #self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          #32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
          output_dims[0], hw[0], strides[0], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
          output_dims[1], hw[1], strides[1], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
          output_dims[2], hw[2], strides[2], initializer, activation_fn, self.cnn_format, name='l3')

      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        hs = hidsize[0]
        self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
          linear(self.l3_flat, hs, activation_fn=activation_fn, name='value_hid')
        self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
          linear(self.l3_flat, hs, activation_fn=activation_fn, name='adv_hid')

        self.value, self.w['val_w_out'], self.w['val_w_b'] = \
          linear(self.value_hid, 1, name='value_out')

        self.advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
          linear(self.adv_hid, self.env.action_size, name='adv_out')

        # Average Dueling
        self.q = self.value + (self.advantage - 
          tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, hidsize[0], activation_fn=activation_fn, name='l4')
        lh = self.l4
        self.q, self.w['q_w'], self.w['q_b'] = linear(lh, self.env.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=1)

      q_summary = []
      avg_q = tf.reduce_mean(self.q, 0)
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.merge_summary(q_summary, 'q_summary')

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t, 
          output_dims[0], hw[0], strides[0], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
          output_dims[1], hw[1], strides[1], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
          output_dims[2], hw[2], strides[2], initializer, activation_fn, self.cnn_format, name='target_l3')

      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

      if self.dueling:
        self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
          linear(self.target_l3_flat, hs, activation_fn=activation_fn, name='target_value_hid')
        self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
          linear(self.target_l3_flat, hs, activation_fn=activation_fn, name='target_adv_hid')

        self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
          linear(self.t_value_hid, 1, name='target_value_out')

        self.t_advantage, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
          linear(self.t_adv_hid, self.env.action_size, name='target_adv_out')

        # Average Dueling
        self.target_q = self.t_value + (self.t_advantage - 
          tf.reduce_mean(self.t_advantage, reduction_indices=1, keep_dims=True))
      else:
        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
            linear(self.target_l3_flat, hidsize[0], activation_fn=activation_fn, name='target_l4')
        lh = self.target_l4
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
            linear(lh, self.env.action_size, name='target_q')

      self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
      self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
      self.action = tf.placeholder('int64', [None], name='action')

      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

      self.delta = self.target_q_t - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      
      #self.gradients_in_epoch = tf.placeholder('float32', None, name='gradients_in_epoch')      
      #self.gradient_mag_op= [tf.reduce_mean([tf.nn.l2_loss() for gradient in _])
                             #for _ in _]
      self.sgd_rule= tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01)
      #self.sgd_rule= tf.train.AdamOptimizer(learning_rate=self.learning_rate_opt)
      self.optim = self.sgd_rule.minimize(self.loss)
      
      gm_summary = []
      for grad, var in self.sgd_rule.compute_gradients(self.loss):
        if grad is not None:
          gm_summary.append(tf.histogram_summary('GM/%s'%var.name, tf.nn.l2_loss(grad)))
      self.gm_summary = tf.merge_summary(gm_summary, 'gm_summary')      
      
      halt= True

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions','training.gradient_mags']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    if self.load_weights:
      self.load_model()
    else:
      import pdb; pdb.set_trace()
      print "WARNING: NOT LOADING WEIGHTS."
    self.wp = {k:v.eval(self.sess) for k, v in self.w.iteritems()}
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)
    if self.folder_name not in os.listdir('./txt/'):
      os.mkdir('./txt/{}'.format(self.folder_name))
    with open("./txt/{}/epochs.txt".format(self.folder_name), mode='a') as f:
      s = "{step} {e_avg_r} {e_max_r} {e_min_r} {e_num_g} {avg_l} {avg_q} {avg_r} \n".format(
        step = self.step,
        e_avg_r = tag_dict["episode.avg reward"],
        e_max_r = tag_dict["episode.max reward"],
        e_min_r = tag_dict["episode.min reward"],
        e_num_g = tag_dict["episode.num of game"],
        avg_l = tag_dict["average.loss"],
        avg_q = tag_dict["average.q"],
        avg_r = tag_dict["average.reward"]
      )
      f.write(s)

  def play(self, experiment, data_dir, n_step=10000, n_episode=2500, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    winners, losers = [], []
    steps_min, steps_taken = [], []
    
    CMAT = np.zeros((len(self.env.shapes),len(self.env.shapes)))
    actions_after_grab = []
    for idx in tqdm(range(n_episode), ncols=70):
      screen = self.env.reset()

      for _ in range(self.history_length):
        test_history.add(screen)

      #for t in tqdm(range(n_step), ncols=70):
      for t in range(n_step):
        # 1. predict
        action = self.predict(test_history.get(), test_ep)
        # 2. act
        screen, reward, terminal, info = self.env.step(action)
        # 3. observe
        test_history.add(screen)
        
        if self.display:
          self.env.render()        

        if terminal:
          winners.append(info['winner'])          
          if experiment == "preference":
            losers.append(info['loser'])
            CMAT[info['winner'],info['loser']] += 1
          if experiment == "one_block":
            steps_min.append(info['n_steps_min'])
            steps_taken.append(info['n_steps'])
            actions_after_grab.append(info['actions_after_grab'])
          break

    #import pdb; pdb.set_trace()
    if experiment == "preference":
      print("Winners: ")
      print(Counter(winners))
      print("Losers: ")
      print(Counter(losers))
      
      with h5py.File("preference_mat.h5","a") as hf:
        hf.create_dataset("C",data=CMAT)      
      
    if experiment == "one_block":
      stats = np.column_stack([np.array(winners),
                              np.array(steps_min),
                              np.array(steps_taken)])
      actions_after_grab = np.array([np.array(a) for a in actions_after_grab])
      diff = [c - b for (a, b, c) in stats]
      
      one_block_saver = Saver(path='{}/{}'.format(data_dir,'one_block_results'))
      D1 = {"stats":stats}
      D2 = {"actions_after_grab": {str(k) : v for (k,v) in zip(range(len(actions_after_grab)),
                                                          actions_after_grab)}}
      one_block_saver.save_dict(0, D1, name="stats")
      #one_block_saver.save_dict(0, D2, name="actions_after_grab")
      one_block_saver.save_recursive_dict(0, D2, name="actions_after_grab")
      
      #one_block_saver.save_dict(0, stats, name="stats")
      #one_block_saver.save_dict(0, actions_after_grab, name="actions_after_grab")
      
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
