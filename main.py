import numpy as np
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state

def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    
    image.save(filename)

if __name__ == '__main__':

    env = UAVEnv(num_uav=3, num_target=5)
    # print(env.info)
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2
    num_uav = 3
    num_target = 5
    # 调试部分
    # 打印观察空间维度进行调试
    print("Observation space dimensions:")
    for agent_id in env.observation_space.keys():
        print(f"{agent_id}: {env.observation_space[agent_id].shape[0]}")
    print(f"Total critic dimensions: {critic_dims}")

    maddpg_agents = MADDPG(actor_dims=actor_dims, critic_dims=critic_dims,n_agents=8, n_actions=n_actions,
                           num_uav=3,num_target=5, 
                           fc1=128, fc2=128,
                           alpha=0.00001, beta=0.02, gamma =0.7, scenario='UAV_Round_up_3uav_5target',
                           chkpt_dir='tmp/maddpg_3uav_5target/')

    memory = MultiAgentReplayBuffer(2000000, critic_dims=critic_dims, actor_dims=actor_dims, 
                        n_actions=n_actions, batch_size=256)

    PRINT_INTERVAL = 100
    N_GAMES = 5000
    MAX_STEPS = 150
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False # True for 验证, False for 训练
    best_score = -50 

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')
    
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = False
        episode_step = 0
        while not dones: # 当所有目标点都被覆盖时结束
            if evaluate:
                # env.render()
                env_render = env.render()
                if episode_step % 10 == 0:
                    # Save the image every 10 episode steps
                    filename = f'images/episode_{i}_step_{episode_step}.png'
                    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if it doesn't exist
                    save_image(env_render, filename)
                # time.sleep(0.01)
            actions = maddpg_agents.choose_action(obs,total_steps,evaluate)
            obs_, rewards, dones = env.step(actions)

            done = dones[0]

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = True 

            memory.store_transition(obs, state, actions, rewards, obs_, state_, [done]*n_agents)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory,total_steps)

            obs = obs_
            score += sum(rewards[0: num_uav])
            score_target += sum(rewards[num_uav:])
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score',avg_score ,'>', best_score, 'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score),'; average target score {:.1f}'.format(avg_target_score))
    
    # save data
    file_name = 'score_history_3uav_5target.csv'
    if not os.path.exists(file_name):
        header = ['episode', 'uav1_score', 'uav2_score', 'uav3_score', 
              'target1_score', 'target2_score', 'target3_score', 
              'target4_score', 'target5_score']
        df = pd.DataFrame([score_history], columns=header)
        df.to_csv(file_name, index=False)
        #pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)