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
    maddpg_agents = MADDPG(actor_dims=actor_dims, critic_dims=critic_dims,n_agents=8, n_actions=n_actions,
                           num_uav=3,num_target=5, 
                           fc1=128, fc2=128,
                           alpha=0.0000025, beta=0.0005, gamma =0.95, scenario='UAV_Round_up_3uav_5target',
                           chkpt_dir='tmp/maddpg_3uav_5target/')

    memory = MultiAgentReplayBuffer(2000000, critic_dims=critic_dims, actor_dims=actor_dims, 
                        n_actions=n_actions, batch_size=512)

    PRINT_INTERVAL = 100
    N_GAMES = 8000
    MAX_STEPS = 150
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = True # True for 验证, False for 训练
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
        while not dones: # 当所有任务完成时结束
            if evaluate: # 评估
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

            if total_steps % 5 == 0 and not evaluate:
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
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            #print(f'Episode {i}:')
            #print(f'Average score: {avg_score:.1f}')
    
    # save data
    file_name = 'score_history_3uav_5target.csv'
    header = ['episode', 'uav1_score', 'uav2_score', 'uav3_score', 
          'target1_score', 'target2_score', 'target3_score', 
          'target4_score', 'target5_score']

    # 创建每个智能体的分数列表
    episode_scores = {
        'episode': i,
        'uav1_score': score_history[-1] if len(score_history) > 0 else 0,
        'uav2_score': score_history[-1] if len(score_history) > 0 else 0,
        'uav3_score': score_history[-1] if len(score_history) > 0 else 0,
        'target1_score': target_score_history[-1] if len(target_score_history) > 0 else 0,
        'target2_score': target_score_history[-1] if len(target_score_history) > 0 else 0,
        'target3_score': target_score_history[-1] if len(target_score_history) > 0 else 0,
        'target4_score': target_score_history[-1] if len(target_score_history) > 0 else 0,
        'target5_score': target_score_history[-1] if len(target_score_history) > 0 else 0
    }
    if not os.path.exists(file_name):
        df = pd.DataFrame([episode_scores])
        df.to_csv(file_name, index=False)
        #pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        df = pd.DataFrame([episode_scores])
        df.to_csv(file_name, mode='a', header=False, index=False)
    print(f"Training complete. Data saved to {file_name}")