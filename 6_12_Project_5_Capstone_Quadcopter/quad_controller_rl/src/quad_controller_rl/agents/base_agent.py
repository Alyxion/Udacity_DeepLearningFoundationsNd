"""Policy search agent."""

import numpy as np
import os
import pandas as pd
from quad_controller_rl import util

from quad_controller_rl.agents.replay_buffer import ReplayBuffer

class BaseAgent:
    """Advanced base agent that lets you limit the action and state space"""
    def __init__(self, task, action_min, action_max, state_min, state_max):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        
        self.min_action = action_min # define minimum and maximum action
        self.max_action = action_max
        
        self.min_stat = state_min # define minimum and maximum state
        self.max_stat = state_max
        
        self.learn_when_done = False # defines if the agent shall only learn at the end of each episode

        # Constrain state and action spaces
        self.state_size = self.max_stat-self.min_stat+1  # position only
        self.action_size = self.max_action-self.min_action+1  # force only
        print("Original spaces: {}, {}\nConstrained spaces: {}, {}".format(
            self.task.observation_space.shape, self.task.action_space.shape,
            self.state_size, self.action_size))
        
        # calc state space minimum and range
        self.state_low = self.task.observation_space.low[self.min_stat:self.max_stat+1]
        self.state_range = self.task.observation_space.high[self.min_stat:self.max_stat+1] - self.state_low
        # self.action_size = np.prod(self.task.action_space.shape)
        
        # calc action space minimum, maximum and range
        self.action_low = self.task.action_space.low[self.min_action:self.max_action+1]
        self.action_high = self.task.action_space.high[self.min_action:self.max_action+1]
        self.action_range = self.action_high-self.action_low

        # Replay memory
        self.epsilon = 0.0
        self.batch_size = 64
        self.buffer_size = 100000
        self.memory = ReplayBuffer(self.buffer_size)

        # Save episode stats
        self.stats_filename = os.path.join(
            util.get_param('out'),
            "stats_{}.csv".format(util.get_timestamp()))  # path to CSV file
        self.stats_columns = ['episode', 'total_reward', 'learning_rate']  # specify columns to save
        self.episode_num = 1
        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))  # [debug]        

        # Episode variables
        self.reset_episode_vars()
        
    def reset_episode_vars(self):
        """Reset current episode's stats"""
        self.last_state = None
        self.org_last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[self.min_stat:self.max_stat+1]  # limit to desired state range

    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[self.min_action:self.max_action+1] = action  # extend to original size again
        return complete_action        
    
    def handle_step_index(self, done):
        """Is called once each turn for periodic events"""
        pass

    def step(self, state, reward, done):
        """Handles a single step:
           - Convert input state to simpler one
           - Estimate best action
           - Learn all x rounds
           - Write stats to log
           - Convert internal to external action and return it"""
        
        org_state = state;
                
        # print("Shape: {}".format(state.shape))
        # Transform state vector
        state = self.preprocess_state(state)

        # print("PP Shape: {}".format(state.shape))

        # print("{} {} {}".format(state.shape, self.state_low.shape, self.state_range.shape))
        state = (state - self.state_low) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector
        
        # Choose an action
        action = self.act(state)
        
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            # print("Action shape {}".format(self.last_action.shape))
            
            if len(self.memory)==self.batch_size-1:
                print("Buffer filled, starting learning")
            self.memory.add(self.last_state, self.last_action, reward, state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and (self.learn_when_done==False or done==True):
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        
        # Sum rewards
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1
            
        # convert action from restricted to full space again
        pp_action = self.postprocess_action(action)

        # Learn, if at end of episode
        if done:
            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward, self.epsilon])
            print("Reward: {} Exploration rate: {}".format(self.total_reward, self.epsilon))
            self.episode_num += 1
            self.reset_episode_vars()
            
        # remember this round's data
        self.last_state = state
        self.org_last_state = org_state
        self.last_action = action                
        
        # notify high level handler
        self.handle_step_index(done)
        
        return pp_action
        
    def write_stats(self, stats):
        """Write single episode stats to CSV file."""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)  # single-row dataframe
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))  # write header first time only                

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        pass

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        pass

