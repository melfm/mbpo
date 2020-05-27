import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device=None, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = None
		if device is not None:
			self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size=256):
		ind = np.random.randint(0, self.size, size=batch_size)
		# Device lets you switch running code from either torch or tf
		if self.device:
			return (
				torch.FloatTensor(self.state[ind]).to(self.device),
				torch.FloatTensor(self.action[ind]).to(self.device),
				torch.FloatTensor(self.next_state[ind]).to(self.device),
				torch.FloatTensor(self.reward[ind]).to(self.device),
				torch.FloatTensor(self.not_done[ind]).to(self.device)
			)
		else:
			return dict(obs1=self.state[ind],
						acts=self.action[ind],
						obs2=self.next_state[ind],
						rews=self.reward[ind],
						done=self.not_done[ind])


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

	def load_yuchen_demo(self, save_folder, size=-1):

		demo_data = np.load(f"{save_folder}.npz")
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		# The data is trajectory per demo so (40, 40, x)
		# The buffer expects (flattened_trajectories, x)
		self.size = min(demo_data["r"].shape[0] * demo_data["r"].shape[1] , size)
		# The state obs in the demo data has extra entry dimension, why? next_state?
		self.size_obs = min(demo_data["o"].shape[0] * demo_data["o"].shape[1] , size)
		# Flatten these
		self.state[:self.size_obs] = demo_data["o"].reshape(-1, demo_data["o"].shape[-1])
		self.action[:self.size] = demo_data["u"].reshape(-1, demo_data["u"].shape[-1])
		# For training the potential, this does not matter as its not used
		# self.next_state[:self.size_obs] = demo_data["o"].reshape(-1, demo_data["o"].shape[-1]) 
		self.reward[:self.size] = demo_data["r"].reshape(-1, demo_data["r"].shape[-1])
		self.not_done[:self.size] = demo_data["done"].reshape(-1, demo_data["done"].shape[-1])
