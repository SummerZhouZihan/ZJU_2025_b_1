import os
import torch as T
import torch.nn.functional as F
from agent import Agent
import numpy as np
# from torch.utils.tensorboard import SummaryWriter

## 所有的智能体共享相同的网络结构和学习参数, 但每个智能体使用不同的状态维度(actor_dims)进行训练
class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, num_uav=3, num_target=5,
                 scenario='simple',  alpha=0.00001, beta=0.02, fc1=128, 
                 fc2=128, gamma=0.7, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = num_uav + num_target
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.num_uav = num_uav
        self.num_target = num_target

        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))
        # 分别为UAV和目标创建Agent
        # 创建UAV智能体
        for uav_idx in range(self.num_uav):
            agent = Agent(
                actor_dims=actor_dims[uav_idx],
                critic_dims=critic_dims + (num_uav + num_target) * n_actions,
                n_actions=n_actions,
                n_agents=self.num_uav +self.num_target,
                agent_idx=uav_idx,
                alpha=alpha,
                beta=beta,
                chkpt_dir=chkpt_dir + '/uav_' + str(uav_idx),
                fc1=fc1,
                fc2=fc2
            )
            self.agents.append(agent)
        
        # 创建target智能体
        for target_idx in range(self.num_target):
            agent = Agent(
                actor_dims=actor_dims[self.num_uav + target_idx],
                critic_dims=critic_dims + (num_uav + num_target) * n_actions,
                n_actions=n_actions,
                n_agents=self.num_target + self.num_uav,
                agent_idx=self.num_uav ,
                alpha=alpha,
                beta=beta,
                chkpt_dir=chkpt_dir + '/target_' + str(target_idx),
                fc1=fc1,
                fc2=fc2
            )
            self.agents.append(agent)


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # 保存UAV模型
        for uav_idx in range(self.num_uav):
            os.makedirs(os.path.dirname(self.agents[uav_idx].actor.chkpt_file), exist_ok=True)
            self.agents[uav_idx].save_models()
    
        # 保存目标模型
        for target_idx in range(self.num_target):
            agent_idx = self.num_uav + target_idx
            os.makedirs(os.path.dirname(self.agents[agent_idx].actor.chkpt_file), exist_ok=True)
            self.agents[agent_idx].save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        # 加载UAV模型
        for uav_idx in range(self.num_uav):
            self.agents[uav_idx].load_models()
    
        # 加载目标模型
        for target_idx in range(self.num_target):
            agent_idx = self.num_uav + target_idx
            self.agents[agent_idx].load_models()

    def choose_action(self, raw_obs, time_step, evaluate):# timestep for exploration
        actions = []
        # 处理UAV的动作
        for uav_idx in range(self.num_uav):
            action = self.agents[uav_idx].choose_action(raw_obs[uav_idx], time_step, evaluate)
            actions.append(action)

        # 处理目标的动作
        for target_idx in range(self.num_target):
            actions.append(np.zeros(2)) # 目标不移动
        # 验证动作列表长度
        expected_length = self.num_uav + self.num_target
        assert len(actions) == expected_length, f"动作列表长度错误: 期望{expected_length}, 实际{len(actions)}"
        return actions

    def learn(self, memory, total_steps):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []
    
        # 分别处理UAV和目标的新动作
        for uav_idx in range(self.num_uav):
            new_states = T.tensor(actor_new_states[uav_idx], dtype=T.float).to(device)
            new_pi = self.agents[uav_idx].target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[uav_idx])
    
        for target_idx in range(self.num_target):
            agent_idx = self.num_uav + target_idx
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_pi = self.agents[agent_idx].target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:,agent_idx] + (1-dones[:,0].int())*agent.gamma*critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            oa = old_actions.clone()
            oa[:,agent_idx*self.n_actions:agent_idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)            
            actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.actor.scheduler.step()

            # self.writer.add_scalar(f'Agent_{agent_idx}/Actor_Loss', actor_loss.item(), total_steps)
            # self.writer.add_scalar(f'Agent_{agent_idx}/Critic_Loss', critic_loss.item(), total_steps)

            # for name, param in agent.actor.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Actor_Gradients/{name}', param.grad, total_steps)
            # for name, param in agent.critic.named_parameters():
            #     if param.grad is not None:
            #         self.writer.add_histogram(f'Agent_{agent_idx}/Critic_Gradients/{name}', param.grad, total_steps)
            
        for agent in self.agents:    
            agent.update_network_parameters()
