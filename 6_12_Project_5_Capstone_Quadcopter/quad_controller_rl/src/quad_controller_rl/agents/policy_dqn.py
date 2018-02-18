"""Deep Q Learnig agent"""

import numpy as np
from quad_controller_rl.agents.base_agent import BaseAgent
import os
import pandas as pd
import random
from quad_controller_rl import util
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

class DQN_Agent(BaseAgent):
    """Reinforcement Learning agent that learns using DQN.
    
    Base upon Keon's cart pole DQN idea"""
    def __init__(self, task):
        super().__init__(task, 2, 2, 2, 2) # setup using advanced base agent using an action space of 1 and a state space of 1
        
        self.z_index = 2
        # self.z_speed_index = 1 # we reuse the original y index here
        
        discrete_actions = 16 # number of discrete actions, in our case force levels
        
        # minimum force of 10, will sink fast even with 10, all below would anyway lead to crashes
        self.min_force = 10.0
        self.max_force = self.action_high[0]
        # calculate discrete actions from 10 to 25
        stepping = (self.max_force-self.min_force)/discrete_actions
        self.discrete_actions = np.arange(self.min_force+stepping, self.max_force+0.1, stepping)
        self.action_size = len(self.discrete_actions)
        
        print("Actions: {} Count {}".format(self.discrete_actions, self.action_size))
        
        self.gamma = 0.8    # discount rate
        self.epsilon = 1.0  # exploration rate, 100% at the beginning, will quickly decrease over 200 rounds
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.96 # fine foe 200 rounds of training
        self.learning_rate = 0.001
        
        # setup keras model
        self.model = self.setup_model()        
        
        # use a small buffer for our DQN
        self.buffer_size = 200
        self.memory = ReplayBuffer(self.buffer_size)        
        
        # define that we only want to learn at the end of each episode
        self.learn_when_done = True
        
        self.learning = True
        # self.set_hover_mode()
        # self.set_takeoff_mode()
        
    def setup_model(self):
        """Setups the keras model. 
        
        Model input: Our current state
        Model output: The estimated Q values for each possible action"""
        
        print("Model setup")
        
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))        
        return model
    
    def handle_step_index(self, done):
        """Is called once each turn for periodic events"""
        if done and self.episode_num%50 == 0 and self.learning==True: # save weights each 50 episodes
            self.store_weights()
        if done and self.episode_num==200 and self.learning==True: # stop learning at some point and validate weights
            print("Learning process finished")
            self.epsilon = 0.0
            self.learning = False
        
    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        if np.random.rand() <= self.epsilon: # if random value between 0..1 < 1, 0.5, 0.25 etc. it will choose a random action.
            return random.randrange(self.action_size)
        act_values = self.model.predict(states)
        return np.argmax(act_values[0])  # returns action at the index with maximum q value in current state
    
    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        
        new_state = state
        # new_state[self.z_speed_index] = 0.0 # we use the original y data index here to store our z velocity
        # if self.org_last_state is not None:
        #     new_state[self.z_speed_index] = self.org_last_state[self.z_index]-state[self.z_index]
            
        # print(new_state)
        
        return new_state[self.min_stat:self.max_stat+1]  # limit to desired state range    
    
    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)  # shape: (6,)
        complete_action[self.z_index] = self.discrete_actions[action]
        
        # print("{} {}".format(complete_action, self.epsilon))
        
        return complete_action    
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # for all experience tuples, replay them now...
        # Note field_names=["state", "action", "reward", "next_state", "done"]), see replay_buffer.py
        if self.learning==False:
            return
        
        for e in experiences: # for all experiences...
            target = e.reward
            if not e.done:
                # reward is known reward + discounted, estimated future reward
                target = e.reward + self.gamma * np.amax(self.model.predict(e.next_state)[0])                
            
            # calc q values from start state
            target_f = self.model.predict(e.state)
            target_f[0][e.action] = target
            # fit model to estimate Q values more precisely
            self.model.fit(e.state, target_f, epochs=1, verbose=0)
        
        # slowly decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def restore_weights(self, filename):
        """Restores weights from given file name in the log folder"""
        weights_file = os.path.join(util.get_param('out'),filename)
        print("Loading weights file from {}".format(weights_file));
        self.model.load_weights(weights_file)

    def store_weights(self):        
        """Stores weights in given file name in the log folder"""
        weights_file = os.path.join(util.get_param('out'),"weights.hdf5")
        print("Storing weight backup in {}".format(weights_file));
        self.model.save_weights(weights_file)
        
    def set_hover_mode(self):
        """Initializes weights base on backuped hover mode"""
        self.learning = False
        self.restore_weights("weights_hover.hdf5")
        self.epsilon = 0.0

    def set_takeoff_mode(self):
        """Initializes weights base on backuped takeoff mode"""
        self.learning = False
        self.restore_weights("weights_takeoff.hdf5")
        self.epsilon = 0.0

    def set_land_mode(self):
        """Initializes weights base on backuped land mode"""
        self.learning = False
        self.restore_weights("weights_land.hdf5")
        self.epsilon = 0.0
