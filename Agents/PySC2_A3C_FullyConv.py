"""
PySC2_A3C_AtariNetNew.py
A script for training and running an A3C agent on the PySC2 environment, with reference to DeepMind's paper:
[1] Vinyals, Oriol, et al. "Starcraft II: A new challenge for reinforcement learning." arXiv preprint arXiv:1708.04782 (2017).
Advantage estimation uses generalized advantage estimation from:
[2] Schulman, John, et al. "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

Credit goes to Arthur Juliani for providing for reference an implementation of A3C for the VizDoom environment
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
https://github.com/awjuliani/DeepRL-Agents

This follows the AtariNet implementation described in [1]. 
The agent takes as input all of the features and outputs a policy across all 524 actions, which makes it generalizable to any of the minigames supplied in SC2LE.
"""

import threading
import multiprocessing
import psutil
import numpy as np
import tensorflow as tf
import scipy.signal
from time import sleep
import os
import json
import pickle

from pysc2.env import sc2_env
from pysc2.env import environment
from pysc2.lib import actions
from pysc2.maps import mini_games

"""
Use the following command to launch Tensorboard:
tensorboard --logdir=worker_0:'./train_0',worker_1:'./train_1',worker_2:'./train_2',worker_3:'./train_3'
"""

## HELPER FUNCTIONS

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	op_holder = []
	for from_var,to_var in zip(from_vars,to_vars):
		op_holder.append(to_var.assign(from_var))
	return op_holder

# Processes PySC2 observations
def process_observation(observation, action_spec, observation_spec):
	# is episode over?
	episode_end = observation.step_type == environment.StepType.LAST
	# reward
	reward = observation.reward
	# features
	features = observation.observation
	variable_features = ['cargo', 'multi_select', 'build_queue']
	max_no = {'available_actions': len(action_spec.functions), 'cargo': 100, 'multi_select': 100, 'build_queue': 10}
	# nonspatial features
	nonspatial_stack = []
	nonspatial_stack = np.log(features['player'].reshape(-1) + 1.)
	nonspatial_stack = np.concatenate((nonspatial_stack, features['game_loop'].reshape(-1)))
	nonspatial_stack = np.expand_dims(nonspatial_stack, axis=0)
	# spatial_minimap features
	minimap_stack = np.stack((features['minimap']), axis=2)
	minimap_stack = np.expand_dims(minimap_stack, axis=0)
	# spatial_screen features
	screen_stack = np.stack((features['screen']), axis=2)
	screen_stack = np.expand_dims(screen_stack, axis=0)
	return reward, nonspatial_stack, minimap_stack, screen_stack, episode_end

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

def sample_dist(dist):
	sample = np.random.choice(dist[0],p=dist[0])
	sample = np.argmax(dist == sample)
	return sample

## ACTOR-CRITIC NETWORK

class AC_Network():
	def __init__(self, scope, trainer, action_spec, observation_spec):
		with tf.variable_scope(scope):
			# Architecture here follows Atari-net Agent described in [1] Section 4.3
			nonspatial_size = 12
			minimap_channels = 7
			screen_channels = 17

			self.inputs_nonspatial = tf.placeholder(shape=[None,nonspatial_size], dtype=tf.float32)
			self.inputs_spatial_minimap = tf.placeholder(shape=[None,64,64,minimap_channels], dtype=tf.float32)
			self.inputs_spatial_screen = tf.placeholder(shape=[None,64,64,screen_channels], dtype=tf.float32)
			self.nonspatial_dense = tf.layers.dense(
				inputs=self.inputs_nonspatial,
				units=32,
				activation=tf.tanh)
			self.screen_conv1 = tf.layers.conv2d(
				inputs=self.inputs_spatial_screen,
				filters=16,
				kernel_size=[5,5],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu)
			self.screen_conv2 = tf.layers.conv2d(
				inputs=self.screen_conv1,
				filters=32,
				kernel_size=[3,3],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu)
			self.minimap_conv1 = tf.layers.conv2d(
				inputs=self.inputs_spatial_minimap,
				filters=16,
				kernel_size=[5,5],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu)
			self.minimap_conv2 = tf.layers.conv2d(
				inputs=self.minimap_conv1,
				filters=32,
				kernel_size=[3,3],
				strides=[1,1],
				padding='same',
				activation=tf.nn.relu)
			screen_output_length = 1
			for dim in self.screen_conv2.get_shape().as_list()[1:]:
				screen_output_length *= dim
			minimap_output_length = 1
			for dim in self.minimap_conv2.get_shape().as_list()[1:]:
				minimap_output_length *= dim
			self.latent_vector_nonspatial = tf.layers.dense(
				inputs=tf.concat([self.nonspatial_dense, tf.reshape(self.screen_conv2,shape=[-1,screen_output_length]), tf.reshape(self.minimap_conv2,shape=[-1,minimap_output_length])], axis=1),
				units=256,
				activation=tf.nn.relu)
			
			# Output layers for policy and value estimations
			# 12 policy networks for base actions and arguments
			#   - All modeled independently
			#   - Spatial arguments have the x and y values modeled independently as well
			# 1 value network
			spatial_arguments = ['screen', 'minimap', 'screen2']
			self.policy_base_actions = tf.layers.dense(
				inputs=self.latent_vector_nonspatial,
				units=len(action_spec.functions),
				activation=tf.nn.softmax,
				kernel_initializer=normalized_columns_initializer(0.01))
			self.policy_arg_nonspatial = dict()
			for arg in action_spec.types:
				if arg.name not in spatial_arguments:
					self.policy_arg_nonspatial[arg.name] = dict()
					for dim, size in enumerate(arg.sizes):
						if size == 2:
							self.policy_arg_nonspatial[arg.name][dim] = tf.layers.dense(
								inputs=self.latent_vector_nonspatial,
								units=size,
								activation=tf.nn.softmax,
								kernel_initializer=normalized_columns_initializer(1.0))
						else:
							self.policy_arg_nonspatial[arg.name][dim] = tf.layers.dense(
								inputs=self.latent_vector_nonspatial,
								units=size,
								activation=tf.nn.softmax,
								kernel_initializer=normalized_columns_initializer(0.01))
			self.policy_arg_spatial = dict()
			self.latent_vector_spatial = dict()
			for arg in spatial_arguments:
				self.latent_vector_spatial[arg] = tf.layers.conv2d(
					inputs=tf.concat([self.screen_conv2, self.minimap_conv2], axis=3),
					filters=1,
					kernel_size=[1,1],
					strides=[1,1],
					padding='same',
					activation=None)
				self.policy_arg_spatial[arg] = tf.nn.softmax(tf.reshape(self.latent_vector_spatial[arg], shape=[-1, 64 * 64]))
			self.value = tf.layers.dense(
				inputs=self.latent_vector_nonspatial,
				units=1,
				kernel_initializer=normalized_columns_initializer(1.0))
			# Only the worker network need ops for loss functions and gradient updating.
			# calculates the losses
			# self.gradients - gradients of loss wrt local_vars
			# applies the gradients to update the global network
			if scope != 'global':
				self.actions_base = tf.placeholder(shape=[None], dtype=tf.int32)
				self.actions_onehot_base = tf.one_hot(self.actions_base, 524, dtype=tf.float32)
				self.actions_arg = dict()
				self.actions_onehot_arg = dict()
				for arg in action_spec.types:
					if arg.name not in spatial_arguments:
						arg_name = arg.name
						self.actions_arg[arg_name] = dict()
						self.actions_onehot_arg[arg_name] = dict()
						for dim, size in enumerate(arg.sizes):
							self.actions_arg[arg_name][dim] = tf.placeholder(shape=[None], dtype=tf.int32)
							self.actions_onehot_arg[arg_name][dim] = tf.one_hot(self.actions_arg[arg_name][dim], size, dtype=tf.float32)
				self.actions_arg_spatial = dict()
				self.actions_onehot_arg_spatial = dict()
				for arg in spatial_arguments:
					self.actions_arg_spatial[arg] = tf.placeholder(shape=[None],dtype=tf.int32)
					self.actions_onehot_arg_spatial[arg] = tf.one_hot(self.actions_arg_spatial[arg], 64 * 64,dtype=tf.float32)
				self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
				self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

				self.responsible_outputs_base = tf.reduce_sum(self.policy_base_actions * self.actions_onehot_base, [1])
				self.responsible_outputs_arg = dict()
				for arg_name in self.policy_arg_nonspatial:
					self.responsible_outputs_arg[arg_name] = dict()
					for dim in self.policy_arg_nonspatial[arg_name]:
						self.responsible_outputs_arg[arg_name][dim] = tf.reduce_sum(self.policy_arg_nonspatial[arg_name][dim] * self.actions_onehot_arg[arg_name][dim], [1])
				self.responsible_outputs_arg_spatial = dict()
				for arg in spatial_arguments:
					self.responsible_outputs_arg_spatial[arg] = tf.reduce_sum(self.policy_arg_spatial[arg] * self.actions_onehot_arg_spatial[arg], [1])

				# Loss functions
				self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))

				self.log_policy_base_actions = tf.log(tf.clip_by_value(self.policy_base_actions, 1e-20, 1.0)) # avoid NaN with clipping when value in policy becomes zero
				self.entropy_base = - tf.reduce_sum(self.policy_base_actions * self.log_policy_base_actions)
				self.entropy_arg = dict()
				for arg_name in self.policy_arg_nonspatial:
					self.entropy_arg[arg_name] = dict()
					for dim in self.policy_arg_nonspatial[arg_name]:
						self.entropy_arg[arg_name][dim] = - tf.reduce_sum(self.policy_arg_nonspatial[arg_name][dim] * tf.log(tf.clip_by_value(self.policy_arg_nonspatial[arg_name][dim], 1e-20, 1.0)))
				self.entropy_arg_spatial = dict()
				for arg in spatial_arguments:
					self.entropy_arg_spatial[arg] = - tf.reduce_sum(self.policy_arg_spatial[arg] * tf.log(tf.clip_by_value(self.policy_arg_spatial[arg], 1e-20, 1.)))
				self.entropy = self.entropy_base
				for arg_name in self.policy_arg_nonspatial:
					for dim in self.policy_arg_nonspatial[arg_name]:
						self.entropy += self.entropy_arg[arg_name][dim]
				for arg in spatial_arguments:
					self.entropy += self.entropy_arg_spatial[arg]

				self.policy_loss_base = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_base, 1e-20, 1.0)) * self.advantages)
				self.policy_loss_arg = dict()
				for arg_name in self.policy_arg_nonspatial:
					self.policy_loss_arg[arg_name] = dict()
					for dim in self.policy_arg_nonspatial[arg_name]:
						self.policy_loss_arg[arg_name][dim] = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_arg[arg_name][dim], 1e-20, 1.0)) * self.advantages)
				self.policy_loss_arg_spatial = dict()
				for arg in spatial_arguments:
					self.policy_loss_arg_spatial[arg] = - tf.reduce_sum(tf.log(tf.clip_by_value(self.responsible_outputs_arg_spatial[arg], 1e-20, 1.0))*self.advantages)
				self.policy_loss = self.policy_loss_base
				for arg_name in self.policy_arg_nonspatial:
					for dim in self.policy_arg_nonspatial[arg_name]:
						self.policy_loss += self.policy_loss_arg[arg_name][dim]
				for arg in spatial_arguments:
					self.policy_loss += self.policy_loss_arg_spatial[arg]

				self.loss = self.value_loss + self.policy_loss - self.entropy * 0.001

				# Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads, self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				
				# Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

## WORKER AGENT

class Worker():
	def __init__(self, name, trainer, model_path, global_episodes, global_steps, map_name, action_spec, observation_spec):
		self.name = "worker_" + str(name)
		self.number = name	
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment_global_episodes = self.global_episodes.assign_add(1)
		self.global_steps = global_steps
		self.increment_global_steps = self.global_steps.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

		self.action_spec = action_spec
		self.observation_spec = observation_spec

		#Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.local_AC = AC_Network(self.name, trainer, action_spec, observation_spec)
		self.update_local_ops = update_target_graph('global', self.name)  
		
		print('Initializing environment #{}...'.format(self.number))
		self.env = sc2_env.SC2Env(map_name=map_name)

		
	def train(self, rollout, sess, gamma, bootstrap_value):
		rollout = np.array(rollout)
		obs_minimap = rollout[:,0]
		obs_screen = rollout[:,1]
		obs_nonspatial = rollout[:,2]
		actions_base = rollout[:,3]
		actions_args = rollout[:,4]
		actions_args_spatial = rollout[:,5]
		rewards = rollout[:,6]
		next_obs_minimap = rollout[:,7]
		next_obs_screen = rollout[:,8]
		next_obs_nonspatial = rollout[:,9]
		values = rollout[:,11]

		actions_arg_stack = dict()
		for actions_arg in actions_args:
			for arg_name in actions_arg:
				if arg_name not in actions_arg_stack:
					actions_arg_stack[arg_name] = dict()
				for dim in actions_arg[arg_name]:
					if dim not in actions_arg_stack[arg_name]:
						actions_arg_stack[arg_name][dim] = [actions_arg[arg_name][dim]]
					else:
						actions_arg_stack[arg_name][dim].append(actions_arg[arg_name][dim])
		actions_arg_spatial_stack = dict()
		for actions_arg_spatial in actions_args_spatial:
			for arg_name,arg_value in actions_arg_spatial.items():
				if arg_name not in actions_arg_spatial_stack:
					actions_arg_spatial_stack[arg_name] = []
				actions_arg_spatial_stack[arg_name].append(arg_value)

		# Here we take the rewards and values from the rollout, and use them to calculate the advantage and discounted returns. 
		# The advantage function uses generalized advantage estimation from [2]
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
		self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
		advantages = discount(advantages,gamma)

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		feed_dict = {self.local_AC.target_v:discounted_rewards,
			self.local_AC.inputs_spatial_screen:np.stack(obs_screen).reshape(-1,64,64,17),
			self.local_AC.inputs_spatial_minimap:np.stack(obs_minimap).reshape(-1,64,64,7),
			self.local_AC.inputs_nonspatial:np.stack(obs_nonspatial).reshape(-1,12),
			self.local_AC.actions_base:actions_base,
			self.local_AC.advantages:advantages}
		for arg_name in actions_arg_stack:
			for dim in actions_arg_stack[arg_name]:
				feed_dict[self.local_AC.actions_arg[arg_name][dim]] = actions_arg_stack[arg_name][dim]
		for arg_name, value in actions_arg_spatial_stack.items():
			feed_dict[self.local_AC.actions_arg_spatial[arg_name]] = value
		v_l,p_l,e_l,g_n,v_n, _ = sess.run([self.local_AC.value_loss,
			self.local_AC.policy_loss,
			self.local_AC.entropy,
			self.local_AC.grad_norms,
			self.local_AC.var_norms,
			self.local_AC.apply_grads],
			feed_dict=feed_dict)
		return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n,v_n
		
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print ("Starting worker " + str(self.number))
		with sess.as_default(), sess.graph.as_default():				 
			while not coord.should_stop():
				# Download copy of parameters from global network
				sess.run(self.update_local_ops)

				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				
				# Start new episode
				obs = self.env.reset()
				episode_frames.append(obs[0])
				reward, nonspatial_stack, minimap_stack, screen_stack, episode_end = process_observation(obs[0], self.action_spec, self.observation_spec)
				s_minimap = minimap_stack
				s_screen = screen_stack
				s_nonspatial = nonspatial_stack
				
				while not episode_end:
					# Take an action using distributions from policy networks' outputs.
					base_action_dist, arg_spatial_dist, arg_nonspatial_dist, v = sess.run([
						self.local_AC.policy_base_actions, 
						self.local_AC.policy_arg_spatial,
						self.local_AC.policy_arg_nonspatial,
						self.local_AC.value],
						feed_dict={
							self.local_AC.inputs_spatial_minimap: minimap_stack,
							self.local_AC.inputs_spatial_screen: screen_stack,
							self.local_AC.inputs_nonspatial: nonspatial_stack})

					# Apply filter to remove unavailable actions and then renormalize
					base_action_dist[0] += 1e-20
					for action_id, action in enumerate(base_action_dist[0]):
						if action_id not in obs[0].observation['available_actions']:
							base_action_dist[0][action_id] = 0.
					base_action_dist[0] /= np.sum(base_action_dist[0])
						
					action_id = sample_dist(base_action_dist)
					
					arg_sample = dict()
					for arg_name in arg_nonspatial_dist:
						arg_sample[arg_name] = dict()
						for dim in arg_nonspatial_dist[arg_name]:
							arg_sample[arg_name][dim] = sample_dist(arg_nonspatial_dist[arg_name][dim])
					arg_sample_spatial = dict()
					arg_sample_spatial_abs = dict()
					for arg in arg_spatial_dist:
						arg_sample_spatial_abs[arg] = sample_dist(arg_spatial_dist[arg])
						arg_sample_spatial[arg] = [arg_sample_spatial_abs[arg] % 64, arg_sample_spatial_abs[arg] / 64]

					arguments = []
					spatial_arguments = ['screen', 'minimap', 'screen2']
					for argument in self.action_spec.functions[action_id].args:
						name = argument.name
						if name not in spatial_arguments:
							argument_value = []
							for dim, size in enumerate(argument.sizes):
								argument_value.append(arg_sample[name][dim])
						else:
							argument_value = arg_sample_spatial[name]
						arguments.append(argument_value)

					# Set unused arguments to -1 so that they won't be updated in the training
					# See documentation for tf.one_hot
					for arg_name, argument in arg_sample.items():
						if arg_name not in self.action_spec.functions[action_id].args:
							for dim in argument:
								arg_sample[arg_name][dim] = -1
					for arg_name, arg in arg_sample_spatial_abs.items():
						if arg_name not in self.action_spec.functions[action_id].args:
							arg_sample_spatial_abs[arg_name] = -1
					
					a = actions.FunctionCall(action_id, arguments)
					obs = self.env.step(actions=[a])
					r, nonspatial_stack, minimap_stack, screen_stack, episode_end = process_observation(obs[0], self.action_spec, self.observation_spec)

					if not episode_end:
						episode_frames.append(obs[0])
						s1_minimap = minimap_stack
						s1_screen = screen_stack
						s1_nonspatial = nonspatial_stack
					else:
						s1_minimap = s_minimap
						s1_screen = s_screen
						s1_nonspatial = s_nonspatial
					
					# Append latest state to buffer
					episode_buffer.append([s_minimap, s_screen, s_nonspatial,action_id,arg_sample,arg_sample_spatial_abs,r,s1_minimap, s1_screen, s1_nonspatial,episode_end,v[0,0]])
					episode_values.append(v[0,0])

					episode_reward += r
					s_minimap = s1_minimap
					s_screen = s1_screen
					s_nonspatial = s1_nonspatial				 
					sess.run(self.increment_global_steps)
					total_steps += 1
					episode_step_count += 1
					
					# If the episode hasn't ended, but the experience buffer is full, then we make an update step using that experience rollout.
					if len(episode_buffer) == 40 and not episode_end and episode_step_count != max_episode_length - 1:
						# Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
						v1 = sess.run(self.local_AC.value, 
							feed_dict={self.local_AC.inputs_spatial_minimap: minimap_stack, self.local_AC.inputs_spatial_screen: screen_stack,self.local_AC.inputs_nonspatial: nonspatial_stack})[0,0]
						v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
						episode_buffer = []
						sess.run(self.update_local_ops)
					if episode_end:
						break
											
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))
				episode_count += 1

				episode_reward = obs[0].observation['score_cumulative'][0]

				global _max_score, _running_avg_score
				if _max_score < episode_reward:
					_max_score = episode_reward
				_running_avg_score = (2.0 / 101) * (episode_reward - _running_avg_score) + _running_avg_score

				print("{} Step #{} Episode #{} Reward: {}".format(self.name, total_steps, episode_count, episode_reward))
				print("Total Steps: {}\tTotal Episodes: {}\tMax Score: {}\tAvg Score: {}".format(sess.run(self.global_steps), sess.run(self.global_episodes), _max_score, _running_avg_score))

				# Update the network using the episode buffer at the end of the episode.
				if len(episode_buffer) != 0:
					v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.)

				if episode_count % 50 == 0 and episode_count != 0:
					if episode_count % 50 == 0 and self.name == 'worker_0':
						saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
						print ("Saved Model")

					mean_reward = np.mean(self.episode_rewards[-50:])
					mean_length = np.mean(self.episode_lengths[-50:])
					mean_value = np.mean(self.episode_mean_values[-50:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				
				sess.run(self.increment_global_episodes)

def main():
	max_episode_length = 300
	gamma = .99 # discount rate for advantage estimation and reward discounting
	load_model = FLAGS.load_model
	model_path = './model'
	map_name = FLAGS.map_name
	assert map_name in mini_games.mini_games

	global _max_score, _running_avg_score
	_max_score = 0
	_running_avg_score = 0

	print('Initializing temporary environment to retrive action_spec...')
	action_spec = sc2_env.SC2Env(map_name=map_name).action_spec()
	print('Initializing temporary environment to retrive observation_spec...')
	observation_spec = sc2_env.SC2Env(map_name=map_name).observation_spec()
	
	tf.reset_default_graph()

	if not os.path.exists(model_path):
		os.makedirs(model_path)

	with tf.device("/cpu:0"): 
		global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
		global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
		trainer = tf.train.AdamOptimizer(learning_rate=3e-5)
		master_network = AC_Network('global', None, action_spec,observation_spec) # Generate global network
		if FLAGS.n_agents < 1:
			num_workers = psutil.cpu_count() # Set workers to number of available CPU threads
		else:
			num_workers = FLAGS.n_agents
		workers = []
		# Create worker classes
		for i in range(num_workers):
			workers.append(Worker(i, trainer, model_path, global_episodes, global_steps, map_name, action_spec, observation_spec))
		saver = tf.train.Saver(max_to_keep=5)

	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		if load_model == True:
			print ('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())
			
		# This is where the asynchronous magic happens.
		# Start the "work" process for each worker in a separate thread.
		worker_threads = []
		for worker in workers:
			worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
			t = threading.Thread(target=(worker_work))
			t.start()
			sleep(0.5)
			sleep(1.5)
			worker_threads.append(t)
		coord.join(worker_threads)

if __name__ == '__main__':
	import sys
	from absl import flags
	flags.DEFINE_string(name="map_name",
		default="DefeatRoaches",
		help="Name of the map/minigame")
	flags.DEFINE_integer(name="n_agents",
		default=0,
		help="Number of agents; passing anything less than 1 will default to number of available CPU threads")
	flags.DEFINE_boolean(name="load_model",
		default=False,
		help="Load a saved model")
	FLAGS = flags.FLAGS
	FLAGS(sys.argv)
	main()
