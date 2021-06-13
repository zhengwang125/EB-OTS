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
        eva.append(env_rl.output(e, 'V'))  # set 'VIS' for visualization
    return eva
            
if __name__ == "__main__":
    # building subtrajectory env
    traj_path = './TrajData/Geolife_out/'
    traj_amount = 6100
    valid_amount = 1000
    J = 2
    k = 5
    D = 6
    constraint = 50
    len_ = 2
    a_size = J 
    s_size = 2
    a_size_ow, s_size_ow, a_size_cw, s_size_cw = J, 2, k, k*len_
    
    #reinforcement_learning
    env_rl = TrajComp_rl(traj_path, traj_amount + valid_amount, a_size_ow, s_size_ow, a_size_cw, s_size_cw, len_, D, constraint)
    err_bounded = 0.0002
    RL_OW = DeepQNetwork_OW(env_rl.n_features_ow, env_rl.n_actions_ow)
    RL_OW.load('./save/RL_OW_'+'your_trained_model'+'.h5')
    RL_CW = DeepQNetwork_CW(env_rl.n_features_cw, env_rl.n_actions_cw)
    RL_CW.load('./save/RL_CW_'+'your_trained_model'+'.h5')
    eva = run_online_dqn(list(range(traj_amount, traj_amount + valid_amount)))
    res = sum(eva) / len(eva)
    print("effectiveness", res)