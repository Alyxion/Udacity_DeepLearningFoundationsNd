"""Policy search agent."""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
import os
import pandas as pd
from quad_controller_rl import util

from quad_controller_rl.agents.ounoise import OUNoise
from quad_controller_rl.agents.ddpg_critic import Critic
from quad_controller_rl.agents.ddpg_actor import Actor
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

class DDPG_Agent(BaseAgent):
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        super().__init__(task, 2, 2, 2, 2)
        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters
        
        self.noise_scale = 0.25
        self.noise_decay = 0.99
        
    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        cur_decay = self.noise_decay**(self.episode_num)
        cur_noise = self.noise.sample()
        scaled_noise = cur_noise*cur_decay*20
        print("{} {} {}".format(actions, cur_noise, scaled_noise))
        return actions+scaled_noise  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        # print("Learning")
        # print(len(experiences))
        states = np.vstack([e.state for e in experiences if e is not None])
        # print(len(states))
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32)
        # print(len(actions))
        actions = actions.reshape(-1, self.action_size)
        # print(len(actions))
        
        # print("States/Actions: {} {}".format(len(states),len(actions)))
        
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

