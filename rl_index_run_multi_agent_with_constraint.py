# -*- utf-8 -*-
from rl_states_drop_points_multi_agent_with_constraint import TrajComp_rl
from rl_brain_multi_agent_with_constraint import DeepQNetwork_OW, DeepQNetwork_CW
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
np.random.seed(1)

def run_online_dqn(elist):
    eva = []
    for e in elist:
        env_rl.reset(e)
        while env_rl.e < env_rl.N:
            observation_ow = env_rl.run_by_skip_value_6(e)
            action_ow = RL_OW.online_act(observation_ow)
            observation_cw = env_rl.step_choose_safe_3(e, action_ow, err_bounded, k, 'V')
            if bool(1 - env_rl.conOpw):
                action_cw = RL_CW.online_act(observation_cw)
                env_rl.step(e, action_cw, 'V')
        if env_rl.simplified_index[-1] != env_rl.N-1:
             env_rl.boundary(e, 'V') 
        eva.append(env_rl.output(e))  # set 'VIS' for visualization
    return eva
        
def run_rl_comp_dqn():
    training = []
    validation = []
    batch_size = 32
    train_op = 10
    check = 999999
    for episode in range(traj_amount):
        print("-----------------",episode,"-----------------" )
        env_rl.reset(episode)
        
        obs_ow = []
        act_ow = []
        transition_ow = []
        
        obs_cw = []
        act_cw = []
        transition_cw = []
        
        while env_rl.e < env_rl.N:
            observation_ow = env_rl.run_by_skip_value_6(episode)
            obs_ow.append(observation_ow)
            action_ow = RL_OW.act(observation_ow)
            observation_cw = env_rl.step_choose_safe_3(episode, action_ow, err_bounded, k, 'T')
            
            if bool(1 - env_rl.conOpw):
                obs_cw.append(observation_cw)
                action_cw = RL_CW.act(observation_cw)
                reward = env_rl.step(episode, action_cw, 'T')
                
                if len(obs_ow) >= 2:
                    transition_ow.append([obs_ow[-2], act_ow[-1], obs_ow[-1], False])
                    if reward is not None:
                        for tr in transition_ow:
                            RL_OW.remember(tr[0], tr[1], reward, tr[2], tr[3])
                        transition_ow.clear()
                    
                if len(obs_cw) >= 2:
                    transition_cw.append([obs_cw[-2], act_cw[-1], obs_cw[-1], False])
                    if reward is not None:
                        for tr in transition_cw:
                            RL_CW.remember(tr[0], tr[1], reward, tr[2], tr[3])
                        transition_cw.clear()
                act_cw.append(action_cw)
                if len(RL_CW.memory) > batch_size:
                    RL_CW.replay(batch_size)
            act_ow.append(action_ow)
            if len(RL_OW.memory) > batch_size and episode % train_op == 0:
                RL_OW.replay(batch_size)

        RL_OW.update_target_model()
        RL_CW.update_target_model()

        if env_rl.simplified_index[-1] != env_rl.N-1:
             reward = env_rl.boundary(episode, 'T')
        
        train_e = env_rl.output(episode, 'T', err_bounded)  # set 'VIS' for visualization
        show = 10
        if episode % show == 0:
            eva = run_online_dqn(list(range(traj_amount, traj_amount + valid_amount)))
            #println()
            res = sum(eva) / len(eva)
            training.append(train_e)
            validation.append(res)
            print('Training compression ration: {}, Validation compression ration: {}'.format(sum(training[-show:])/len(training[-show:]), res))
            if res < check:
                check = res
                RL_OW.save('./save/RL_OW_' + str(check) +'err_'+ str(err_bounded)+'.h5')
                RL_CW.save('./save/RL_CW_' + str(check) +'err_'+ str(err_bounded)+'.h5')
                print('Save model at episode {} with error {}'.format(episode, res))
            print('==>current best model ratio is {} with err_bounded {}'.format(check, err_bounded))
     
if __name__ == "__main__":
    # building subtrajectory env
    traj_path = './TrajData/Geolife_out/'
    traj_amount = 6000
    valid_amount = 100
    J = 2
    k = 5
    D = 6
    constraint = 50
    len_ = 2
    a_size = J 
    s_size = 2
    a_size_ow, s_size_ow, a_size_cw, s_size_cw = J, 2, k, k*len_
    
    for i in [0.0002]:
        #reinforcement_learning
        env_rl = TrajComp_rl(traj_path, traj_amount + valid_amount, a_size_ow, s_size_ow, a_size_cw, s_size_cw, len_, D, constraint)
        err_bounded = i
        RL_OW = DeepQNetwork_OW(env_rl.n_features_ow, env_rl.n_actions_ow)
        RL_CW = DeepQNetwork_CW(env_rl.n_features_cw, env_rl.n_actions_cw)
        run_rl_comp_dqn()