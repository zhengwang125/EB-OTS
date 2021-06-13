# -*- utf-8 -*-
import numpy as np
import data_utils as F
import heapq
import copy
import matplotlib.pyplot as plt
import math
import warnings
import pickle
import random
from sklearn.utils import shuffle
random.seed(1)

warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class RingBuffer(object):
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = size if padding is None else padding
        self.buffer = np.zeros(self.size+self.padding)
        self.counter = 0
        self.flag = True

    def append(self, data):
        """this is an O(n) operation"""
        data = data[-self.padding:]
        n = len(data)
        if self.remaining < n: 
            self.flag = self.compact()
        #print('data', data)
        self.buffer[self.counter+self.size:][:n] = data
        self.counter += n
        return self.flag

    @property
    def remaining(self):
        return self.padding-self.counter
    @property
    def view(self):
        """this is always an O(1) operation"""
        return self.buffer[self.counter:][:self.size]
    def compact(self):
        """
        note: only when this function is called, is an O(size) performance hit incurred,
        and this cost is amortized over the whole padding space
        """
        self.buffer[:self.size] = self.view
        self.counter = 0
        return False

class TrajComp_rl():
    def __init__(self, path, amount, a_size_ow, s_size_ow, a_size_cw, s_size_cw, len_, D, constraint):
        self.n_actions_ow = a_size_ow
        self.n_features_ow = s_size_ow
        self.n_actions_cw = a_size_cw
        self.n_features_cw = s_size_cw
        self.len_ = len_
        self.D = D
        self.constraint = constraint
        self._load(path, amount)
    
    def _load(self, path, amount):
        self.ori_traj_set = []
        for num in range(amount):
            self.ori_traj_set.append(F.to_traj(path + str(num)))
        #self.reward_opt = pickle.load(open('reward_list_partial', 'rb'), encoding='bytes')
        #self.path_opt = pickle.load(open('path_set_partial', 'rb'), encoding='bytes')

    def reset(self, episode):
        self.origin_index = 0
        self.simplified_index = []
        self.simplified_tra = []
        self.simplified_index.append(self.origin_index)
        self.simplified_tra.append(self.ori_traj_set[episode][self.origin_index])
        self.e = self.origin_index + 2
        self.N = len(self.ori_traj_set[episode])
        self.conOpw = True
        #self.observation_index = [self.origin_index + 1]
        #self.observation_container = [0.0]
        
        self.observation_index_skip = [self.origin_index + 1]
        self.observation_container_skip = [0.0, 1.0]
        
        self.last = 0
        self.current = 0
        self.total_drop_pts = 0
        
        self.delays = self.D
        self.reward_rl = np.zeros(self.N)
        self.action_index_pre = 0
        self.accumulate = 0
        self.accumulate_distance = np.sqrt((self.ori_traj_set[episode][self.origin_index + 1][1] - self.ori_traj_set[episode][self.origin_index][1])**2 + 
                                           (self.ori_traj_set[episode][self.origin_index + 1][0] - self.ori_traj_set[episode][self.origin_index][0])**2)
        self.accumulate_index = self.origin_index + 1
        
    def run_by_drop_num(self, episode, err_bounded, k, total_drop = 0):
        if F.sed_op(self.ori_traj_set[episode][self.origin_index:self.e+1]) > err_bounded:
            self.conOpw = False
            # padding observation_index
            self.observation_index = list(range(self.origin_index+1, self.e))[-k:]
            if len(self.observation_index) < k:
                self.observation_index.extend(self.observation_index[-1:]*(k - len(self.observation_index)))
            # padding observation
            observation = list(range(total_drop + 0, total_drop + self.e - self.origin_index -1))[-k:]
            if len(observation) < k:
                observation.extend(observation[-1:]*(k - len(observation)))
            return np.array(observation).reshape(-1, k)
        else:
            self.e += 1
            return []
    
    def states_normalized(self, observation, _len):
        tmp = [max(observation[i::_len]) for i in range(_len)]
        for i in range(len(observation)):
            if tmp[i%_len] !=0:
                observation[i] = observation[i]/tmp[i%_len]
        return observation
            
    def run_by_drop_value(self, episode, err_bounded, k, total_drop = 0):
        anchor_check = F.sed_op(self.ori_traj_set[episode][self.origin_index:self.e+1])
        if anchor_check > err_bounded:
            self.conOpw = False
            # padding observation_index
            self.observation_index = list(range(self.origin_index+1, self.e))[-k:]
            if len(self.observation_index) < k:
                rb = RingBuffer(k)
                while rb.append(self.observation_index):
                    continue
                self.observation_index = rb.view
            # padding observation
            observation = self.observation_container[-self.len_*k:]
            if len(observation) < self.len_*k:
                rb = RingBuffer(self.len_*k)
                while rb.append(observation):
                    continue
                observation = rb.view
            return np.array(observation).reshape(-1, self.len_*k)
        else:
            
            tmp = [anchor_check]#, self.e - self.origin_index
            '''
            self.ori_traj_set[episode][self.origin_index][0], 
            self.ori_traj_set[episode][self.origin_index][1],
            self.ori_traj_set[episode][self.e][0],
            self.ori_traj_set[episode][self.e][1],
            '''
            self.observation_container.extend(tmp)
            self.e += 1
            return []
    
    def run_by_drop_value_2(self, episode, err_bounded, k, total_drop = 0):
        anchor_check = F.sed_op(self.ori_traj_set[episode][self.origin_index:self.e+1])
        if anchor_check > err_bounded:
            if self.e - self.origin_index - 1 < k: #deterministic rule
                self.simplified_index.append(self.e - 1)
                self.simplified_tra.append(self.ori_traj_set[episode][self.e - 1])
                self.origin_index = self.e - 1
                self.e = self.origin_index + 2
                self.observation_container = [0.0] #init , 1.0
                return []
            self.conOpw = False
            self.observation_index = list(range(self.origin_index+1, self.e))[-k:]
            observation = self.observation_container[-self.len_*k:]
            #observation.extend([self.ori_traj_set[episode][self.origin_index][0], self.ori_traj_set[episode][self.origin_index][1]])
            observation = self.states_normalized(observation, self.len_)
            return np.array(observation).reshape(-1, self.len_*k)
        else:
            tmp = [anchor_check]#, self.e - self.origin_index
            '''
            self.ori_traj_set[episode][self.origin_index][0], 
            self.ori_traj_set[episode][self.origin_index][1],
            self.ori_traj_set[episode][self.e][0],
            self.ori_traj_set[episode][self.e][1],
            '''
            self.observation_container.extend(tmp)
            self.e += 1
            return []
    
    def run_by_skip_value_3(self, episode, J):
        for i in range(self.e, self.e+J):
            if i + 1 < len(self.ori_traj_set[episode]):
                self.observation_container.append(F.sed_op([self.ori_traj_set[episode][self.origin_index],
                                     self.ori_traj_set[episode][i],
                                     self.ori_traj_set[episode][i+1]]))
                self.observation_index.append(i)
        self.observation_index = self.observation_index[-J:]
        observation = self.observation_container[-J:]
        if len(self.observation_index) < J:
            self.observation_index.extend(self.observation_index[-1:]*(J - len(self.observation_index)))
            observation.extend(observation[-1:]*(J - len(observation)))
        observation = self.states_normalized(observation, 1)
        observation, self.observation_index = shuffle(np.array(observation).reshape(-1, 1), self.observation_index, random_state=0)
        
        return np.array(observation).reshape(-1, J)
    
    def run_by_skip_value_4(self, episode, J, err_bounded):
        self.err_record = {}
        for i in range(self.e, self.e+J):
            tmp = F.sed_op(self.ori_traj_set[episode][self.origin_index:i+1])            
            self.observation_container.append(tmp)
            self.observation_index.append(i)
            self.err_record[i] = tmp
            
        self.observation_index = self.observation_index[-J:]
        observation = self.observation_container[-J:]
        if len(self.observation_index) < J:
            self.observation_index.extend(self.observation_index[-1:]*(J - len(self.observation_index)))
            observation.extend(observation[-1:]*(J - len(observation)))
        
        observation = self.states_normalized(observation, 1)
        observation, self.observation_index = shuffle(np.array(observation).reshape(-1, 1), self.observation_index, random_state=0)
        
        return np.array(observation).reshape(-1,J)
    
    def run_by_skip_value_5(self, episode):
        observation = []
        ps = self.ori_traj_set[episode][self.origin_index]
        pe = self.ori_traj_set[episode][self.e]
        A = pe[1] - ps[1]
        B = ps[0] - pe[0]
        observation.append(np.sqrt(A * A + B * B))
        observation.append(self.e - self.origin_index)
        return np.array(observation).reshape(-1,2)
    
    def run_by_skip_value_6(self, episode):
        observation = []
        ps = self.ori_traj_set[episode][self.origin_index]
        pe = self.ori_traj_set[episode][self.e]
        p_tmp = self.ori_traj_set[episode][self.accumulate_index]
        A = pe[1] - ps[1]
        B = pe[0] - ps[0]
        C = pe[1] - p_tmp[1]
        D = pe[0] - p_tmp[0]
        tmp = np.sqrt(A * A + B * B)
        self.accumulate_distance += np.sqrt(C * C + D * D)
        observation.append(tmp/self.accumulate_distance)
        observation.append(self.e - self.origin_index)
        self.accumulate_index = self.e
        return np.array(observation).reshape(-1,2)
        
    def reward_1(self, action_index):
        return action_index - self.origin_index - 1
    
    def reward_2(self, action_index, episode):
        self.total_drop_pts += (action_index - self.origin_index - 1)
        self.current = self.total_drop_pts - self.reward_opt[episode][action_index]
        rw = self.current - self.last
        self.last = self.current
        return rw
    
    def reward_3(self, action_index, episode):
        if action_index in self.path_opt[episode]: #traj_amount temporarily
            return 0
        else:
            return -1
    
    def reward_4(self, action_index, episode):
        self.total_drop_pts += (action_index - self.origin_index - 1)
        rw = self.total_drop_pts - self.reward_opt[episode][action_index]
        return rw
    
    def reward_5(self, action_index, episode, kept):
        '''
        for i in range(self.origin_index, action_index):
            if i == self.origin_index:
                self.reward_rl[i] = i - kept
            else:
                self.reward_rl[i] = i - kept - 1
        '''
        self.reward_rl[self.origin_index] = self.origin_index - kept
        self.reward_rl[self.origin_index+1:action_index] = np.arange(self.origin_index - kept, action_index - kept - 1)
        
        a = self.reward_rl[self.origin_index:action_index]
        b = self.reward_opt[episode][self.origin_index:action_index] 
        if action_index == self.N - 1:
            self.reward_rl[-1] = self.N - len(self.simplified_index)    
            a = self.reward_rl[self.origin_index:action_index+1]
            b = self.reward_opt[episode][self.origin_index:action_index+1]    
        return -np.mean(abs(a-b))
    
    def reward_6(self, action_index, episode):
        self.accumulate += (action_index - self.origin_index - 1)
        if action_index in self.path_opt[episode]:
            if self.reward_opt[episode][action_index] == self.reward_opt[episode][self.action_index_pre]:
                ratio = 1.0
            else:
                ratio = self.accumulate/(self.reward_opt[episode][action_index] - self.reward_opt[episode][self.action_index_pre])
            self.action_index_pre = action_index
            self.accumulate = 0
            return ratio, True
        return None, False
    
    def step_choose_safe_3(self, episode, action, err_bounded, k, label='T'): #action 0 drop, 1 keep
        if action == 0:
            action_index = self.e + 1 #skip 1 point
#        elif action == 1:
#            action_index = self.e + 2 #skip 2 points
        else:
            action_index = self.e #no skipping
        self.e = action_index + 1
        
        anchor_check, is_bounded = F.sed_op_is_bounded_with_err(self.ori_traj_set[episode][self.origin_index:action_index+1], err_bounded)

        if not is_bounded:
            if self.delays == 0 or self.e >= self.N - 1 or len(self.observation_index_skip) > self.constraint:
                if len(self.observation_index_skip) == 1:
                    safe = self.observation_index_skip[-1]
                    self.simplified_index.append(safe)
                    self.simplified_tra.append(self.ori_traj_set[episode][safe])
                    self.origin_index = safe
                    self.e = self.origin_index + 2
                    self.observation_container_skip = [0.0, 1.0] #init
                    self.observation_index_skip = [self.origin_index + 1] #init
                    self.delays = self.D
                    return []
                if len(self.observation_index_skip) < k: 
                    self.conOpw = False
                    self.observation_index_skip = self.observation_index_skip[-k:]
                    self.observation_index_skip.extend(self.observation_index_skip[-1:]*(k - len(self.observation_index_skip)))
                    observation = self.observation_container_skip[-self.len_*k:]
                    observation.extend(observation[-self.len_:]*int((self.len_*k - len(observation))/self.len_))
                    observation = self.states_normalized(observation, self.len_)
                    observation, self.observation_index_skip = shuffle(np.array(observation).reshape(-1, self.len_), self.observation_index_skip, random_state=0)
                    return np.array(observation).reshape(-1, self.len_*k)
                else:
                    self.conOpw = False
                    self.observation_index_skip = self.observation_index_skip[-k:]
                    observation = self.observation_container_skip[-self.len_*k:]
                    observation = self.states_normalized(observation, self.len_)
                    observation, self.observation_index_skip = shuffle(np.array(observation).reshape(-1, self.len_), self.observation_index_skip, random_state=0)
                    return np.array(observation).reshape(-1, self.len_*k)

            else:
                self.delays = self.delays - 1
                return []                
        else:
            tmp = [anchor_check, action_index - self.origin_index]
            self.observation_container_skip.extend(tmp)
            self.observation_index_skip.append(action_index)            
            return []
    
    def step_choose_safe_2(self, episode, action, err_bounded, choose_action=-1, label='T'):
        action_index = int(self.observation_index[action])
        
        self.observation_index = [action_index+1]
        self.observation_container = [0.0]
        rw = 0
        self.e = action_index + 2
        
        anchor_check, is_bounded = F.sed_op_is_bounded_with_err(self.ori_traj_set[episode][self.origin_index:action_index+1], err_bounded)

        if not is_bounded or self.e > self.N - 1:

            safe = self.observation_index_skip[choose_action]
            self.simplified_index.append(safe)
            self.simplified_tra.append(self.ori_traj_set[episode][safe])
            if label == 'T':
                rw = self.reward_1(safe)
            self.origin_index = safe
            self.e = self.origin_index + 2
            self.observation_container_skip = [0.0] #init
            self.observation_index_skip = [self.origin_index + 1] #init
            self.delays = self.D 
            return rw                    

        else:
            tmp = [anchor_check, action_index - self.origin_index]#, self.e - self.origin_index
            '''
            self.ori_traj_set[episode][self.origin_index][0], 
            self.ori_traj_set[episode][self.origin_index][1],
            self.ori_traj_set[episode][self.e][0],
            self.ori_traj_set[episode][self.e][1],
            '''
            self.observation_container_skip.extend(tmp)
            self.observation_index_skip.append(action_index)
            return None
    
    def step_choose_safe_1(self, episode, action, err_bounded, choose_action=-1, label='T'):
        action_index = int(self.observation_index[action])
        
        self.observation_index = [action_index+1]
        self.observation_container = [0.0]
        rw = 0
        self.e = action_index + 2
        
        anchor_check = self.err_record[action_index]

        if anchor_check > err_bounded or self.e > self.N - 1:

            safe = self.observation_index_skip[choose_action]
            self.simplified_index.append(safe)
            self.simplified_tra.append(self.ori_traj_set[episode][safe])
            if label == 'T':
                rw = self.reward_1(safe)
            self.origin_index = safe
            self.e = self.origin_index + 2
            self.observation_container_skip = [0.0] #init
            self.observation_index_skip = [self.origin_index + 1] #init
            self.delays = self.D 
            return rw                    

        else:
            tmp = [anchor_check, action_index - self.origin_index]#, self.e - self.origin_index
            '''
            self.ori_traj_set[episode][self.origin_index][0], 
            self.ori_traj_set[episode][self.origin_index][1],
            self.ori_traj_set[episode][self.e][0],
            self.ori_traj_set[episode][self.e][1],
            '''
            self.observation_container_skip.extend(tmp)
            self.observation_index_skip.append(action_index)
            return None
    
    def step(self, episode, choose_action, label='T'):
        rw = 0
        safe = self.observation_index_skip[choose_action]
        self.simplified_index.append(safe)
        self.simplified_tra.append(self.ori_traj_set[episode][safe])
        if label == 'T':
            rw = self.reward_1(safe)
        self.origin_index = safe
        self.e = self.origin_index + 2
        self.observation_container_skip = [0.0, 1.0] #init
        self.observation_index_skip = [self.origin_index + 1] #init
        self.delays = self.D
        self.conOpw = True
        return rw
        
    def boundary(self, episode, label='T'):
        self.simplified_index.append(self.N - 1)
        self.simplified_tra.append(self.ori_traj_set[episode][-1])
        rw = self.reward_1(self.simplified_index[-1])
        #rw = self.reward_2(self.simplified_index[-1], episode)
        #rw = self.reward_3(self.simplified_index[-1], episode)
        #rw = self.reward_4(self.simplified_index[-1], episode)
        #rw = self.reward_5(self.simplified_index[-1], episode, len(self.simplified_index)-2)
        return rw

    def output(self, episode, label = 'T', err_bounded=0.0002):
        if label == 'VIS':
            F.draw(self.ori_traj_set[episode], self.simplified_tra)
        return len(self.simplified_tra) / self.N
