import gym.spaces
import numpy as np
import os
import random

from rlgym.utils.action_parsers import ActionParser

from rlgym.utils.gamestates import GameState

from hwrlai.helpers.functions import make_state_dict
from hwrlai.helpers.functions import get_seconds

from hwrlai.helpers.functions import logit
from hwrlai.helpers.functions import prtit

from hwrlai.info.data import Files

from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class RLDiscreteAction(ActionParser):
    
    def __init__(self):

        super().__init__()  
        
        self.prev_state_data=()
        self.prev_actions=None
        
        self.nn_train = 0
        
        self.nn_model_file_path = Files.file_nn_model
        self.nn_model = self.create_nn_model()
        self.nn_batch_size = 32
        self.nn_memory = []
        self.nn_gamma = 0.95
        self.nn_eps = 1.0
        self.nn_eps_min = 0.00025
        self.nn_eps_decay = 0.999     
        
        self.previous_state_dict = {}
        
        self.last_save_sQ = 0

        self.loaded_prev_a_reward = 0
        self.max_a_reward = 0
        
        self.q_tbl_save_int = 120 # 1800
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiDiscrete([3] * 5 + [2] * 3) 
    
    def get_random_actions(self):
    
        base_action = [0,0,0,0,0,0,0,0]
        
        base_action[0] = random.randint(-1, 1)
        base_action[1] = random.randint(-1, 1)
        base_action[2] = random.randint(-1, 1)
        base_action[3] = random.randint(-1, 1)
        base_action[4] = random.randint(-1, 1)
        base_action[5] = random.randint(0, 1)
        base_action[6] = random.randint(0, 1)
        base_action[7] = random.randint(0, 1)
        
        return tuple(base_action)

    def get_reward(self, state_dict: dict) -> float:
        
        reward = 0

        sDs = state_dict

        cdtb = 1 - (sDs["distance_to_ball"] / 10000)
        bdtg = 1 - (sDs["distance_to_goal"] / 10200)
        ba = sDs["boost_amount"]
        kdr = sDs["kill_death"]
        fv = sDs["forward_velocity"]/2300
        
        cdtb_w = 2.5
        bdtg_w = 1.5
        ba_w = 1.1
        kdr_w = 1.2
        fv_w = 2.0

        reward = (cdtb*cdtb_w) + (bdtg*bdtg_w) + (ba*ba_w) + (kdr*kdr_w) + (fv*fv_w)

        if 1==2:
            print("cdtb",cdtb)
            print("bdtg",bdtg)
            print("ba",ba)
            print("kdr",kdr)
            print("fv",fv)
            print("reward",reward)
            
        if self.loaded_prev_a_reward == 0:

            if os.path.exists(Files.file_ar):            
                with open(Files.file_ar) as j:
                    content = j.readlines()
                    self.max_a_reward = float(content[0].split("=")[1])
                    prtit("loaded previous ar")

            self.loaded_prev_a_reward = 1

        if round(reward,2)  > self.max_a_reward:
            self.max_a_reward = round(reward,2) 
            with open(Files.file_ar, "w") as myfile:
                myfile.write("ar="+str(self.max_a_reward))   

        return round(reward,2)
    
    def create_nn_model(self):
        if os.path.isfile(self.nn_model_file_path):
            model = load_model(self.nn_model_file_path)
            logit("loaded nn_model")
        else:
            model = Sequential()
            model.add(Input(shape=(5,)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(8, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=0.001))
            logit("created nn_model")
        return model

    def train_nn_model(self):
        if len(self.nn_memory) < self.nn_batch_size:
            return
        
        batch = np.array(self.nn_memory)
        
        states = batch[:,:5]
        actions = batch[:,5: -6]
        rewards = batch[:,-1]
        next_states = batch[:,14:]
        
        target = self.nn_model.predict_on_batch(np.array(states))
        next_q = self.nn_model.predict_on_batch(np.array(next_states))
        max_next_q = np.max(next_q, axis=1)
        actions_idx = np.argmax(actions, axis=1)
        target[np.arange(self.nn_batch_size), actions_idx] = rewards + self.nn_gamma * max_next_q
        self.nn_model.fit(np.array(states), target, batch_size=self.nn_batch_size, epochs=1, verbose=0)

        self.nn_memory = []

    def get_nn_new_actions(self, state):
        
        self.nn_train = 0

        numra = np.random.rand()
        if numra > self.nn_eps:
            new_actions =  self.get_random_actions()
        else:
           self.nn_train = 1
           actions = self.nn_model.predict(np.array(state[:5]).reshape(1,5))
           self.train_nn_model()
           new_actions = np.zeros(8)
           new_actions[:5] = np.clip(np.round(actions[:, :5]), -1, 1)
           new_actions[5:] = np.round(actions[:, 5:])
           new_actions[5:] = np.clip(new_actions[5:], 0, 1)    
           
        if self.nn_eps > self.nn_eps_min:
            self.nn_eps = max(self.nn_eps_min, self.nn_eps - (1 - self.nn_eps_decay))              
            
        return new_actions     
    
    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        
        DsD = make_state_dict(state)
        
        state_data =   (
            DsD["distance_to_ball"],
            DsD["distance_to_goal"],
            DsD["boost_amount"],
            DsD["kill_death"],
            DsD["forward_velocity"]
                        )
        
        if len(self.prev_state_data) > 0:
            reward = self.get_reward(self.previous_state_dict)
            self.state_action_reward = [*self.prev_state_data, *self.prev_actions, reward]

            self.nn_memory.append([*self.state_action_reward, *state_data])
            if len(self.nn_memory) > 32:
                del self.nn_memory[0]           
            actions = self.get_nn_new_actions(self.state_action_reward)

            if get_seconds() > self.last_save_sQ:
                
                if self.nn_train  == 1:

                    logit("saving & nn_model...")
                    self.nn_model.save(Files.file_nn_model)
                    self.last_save_sQ = get_seconds() + self.q_tbl_save_int 
  
        else:
            actions = self.get_random_actions()
                                      
        self.prev_state_data = state_data
        self.prev_actions = actions
        self.previous_state_dict = DsD
        
        if type(actions) is tuple: 
            actions = np.array(list(actions))

        return np.array([actions.tolist()])