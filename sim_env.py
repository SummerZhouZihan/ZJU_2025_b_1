import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy

# 注意, num_agents 是总目标数(追捕者 + 目标)

class UAVEnv:
    def __init__(self,length=2,num_obstacle=0,num_uav=3,num_target=5):
        self.length = length # length of boundary
        self.num_obstacle = num_obstacle # number of obstacles
        self.num_uav = num_uav  # 追捕者数量
        self.num_target = num_target  # 目标数量
        num_agents = num_uav + num_target
        self.num_agents = num_uav + num_target  # 总智能体数量
        self.time_step = 0.1 # update time step(单位:秒)
        self.v_max = 0.05 # 追捕者最大速度
        self.v_max_e = 0 # 目标速度
        # self.a_max = 0.04 # 追捕者最大加速度
        # self.a_max_e = 0.05 # 目标最大加速度
        self.L_sensor = 0.2 # 激光传感器的最大探测距离
        self.num_lasers = 16 # 激光射线的数量
        self.d_capture = 0.0001  # 捕获目标的距离阈值，小于这个距离视作捕获
        self.target_detected = [False] * self.num_target  # 每个目标是否被发现
        self.target_detect_count = [0] * self.num_target  # 初始化每个目标被侦查的次数
        self.multi_current_pos = []  # 初始化位置列表
        self.multi_current_vel = []  # 初始化速度列表
        self.uav_takeoff_times = [i* 10 for i in range(num_uav)]  # 每架UAV的起飞时间步数(10步=1秒)
        self.current_step = 0  # 当前时间步
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)] for _ in range(self.num_agents)]
        # self.agents = ['agent_0','agent_1','agent_2','target']
        self.agents = ['uav_' + str(i) for i in range(num_uav)]
        self.agents.extend(['target_' + str(i) for i in range(num_target)])
        self.info = np.random.get_state() # get seed
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)] # 障碍物列表
        self.history_positions = [[] for _ in range(self.num_agents)]
        self.total_distance = [0] * self.num_agents  # 添加总距离记录
        self.last_positions = None  # 记录上一步位置

        self.action_space = {} # 初始化动作空间
        for i in range(self.num_uav):
            self.action_space[f'uav_{i}'] = spaces.Box(low=-self.v_max, high=self.v_max, shape=(2,))
        for i in range(self.num_target):
            self.action_space[f'target_{i}'] = spaces.Box(low=-self.v_max_e, high=self.v_max_e, shape=(2,))

        #self.observation_space = {}        # 初始化观察空间
        #for i in range(self.num_uav):
        #    self.observation_space[f'uav_{i}'] = spaces.Box(low=-np.inf, high=np.inf, shape=(26,))
        #for i in range(self.num_target):
        #    self.observation_space[f'target_{i}'] = spaces.Box(low=-np.inf, high=np.inf, shape=(23,))
        # 动态计算观察维度
        self.observation_space = {}        #初始化观察空间
        uav_obs_dim = 4 + 2*(num_uav-1) + 3*num_target + 16
        target_obs_dim = 18

        for i in range(self.num_uav):
            self.observation_space[f'uav_{i}'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(uav_obs_dim,))
            
        for i in range(self.num_target):
            self.observation_space[f'target_{i}'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(target_obs_dim,))

    def reset(self):
        SEED = random.randint(1,1000)
        random.seed(SEED)
        self.multi_current_pos = [] # 所有无人机的当前位置
        self.multi_current_vel = [] # 所有无人机的当前速度
        self.history_positions = [[] for _ in range(self.num_agents)] # 无人机的历史位置
        self.total_distance = [0] * self.num_uav  # 重置总距离
        self.current_step = 0  # 重置当前时间步
        # 设置起始位置
        start_positions = []
        for _ in range(self.num_uav):        # UAV起始位置, 都从0开始
            start_positions.append(np.array([0.0, 0.0]))
        # 目标起始位置
        target_positions = [
            np.array([1.2, 0.8]),  # target_0
            np.array([0.3, 0.45]), # target_1
            np.array([0.95, 0.2]), # target_2
            np.array([0.6, 1.2]),  # target_3
            np.array([1.5, 0.5])   # target_4
        ]
        for i in range(self.num_target):
            start_positions.append(target_positions[i])
        # 初始化位置和速度
        self.multi_current_pos = []
        self.multi_current_vel = []
        self.target_detected = [False] * self.num_target  # 每个目标是否被发现
        self.target_detect_count = [0] * self.num_target  # 重置每个目标被侦查的次数
        for i in range(self.num_target + self.num_uav):   # 初始化所有目标速度为0
            self.multi_current_pos.append(start_positions[i])
            self.multi_current_vel.append(np.zeros(2))    # 所有智能体初始速度为0
            self.history_positions[i].append(copy.deepcopy(start_positions[i]))

        # update lasers
        IsCollied = self.update_lasers_isCollied_wrapper()
        ## multi_obs is list of agent_obs, state is multi_obs after flattenned
        multi_obs = self.get_multi_obs()
        return multi_obs

    def step(self,actions):
        self.current_step += 1  # 更新当前时间步
        last_d2target = [] # 记录每个无人机到目标的上一步距离
        # print(actions)
        # time.sleep(0.1)

        for i in range(self.num_uav): # 无人机
            if self.current_step < self.uav_takeoff_times[i]:
                # 未到起飞时间的UAV保持在起飞点
                self.multi_current_vel[i] = np.zeros(2)
                continue
            pos = self.multi_current_pos[i]
            # 记录到目标的距离
            target_distances = []
            for j in range(self.num_target):
                target_idx = self.num_uav + j
                pos_target = self.multi_current_pos[target_idx]
                target_distances.append(np.linalg.norm(pos - pos_target))
            last_d2target.append(min(target_distances))  # 使用最近的目标距离

            # 处理动作 - 确保动作维度正确
            action = actions[i]
            if isinstance(action, np.ndarray):
                if action.shape != (2,):
                    # 如果维度不正确，尝试重塑或取前两个值
                    action = action[:2] if len(action) > 2 else np.zeros(2)
            else:
                action = np.array(action, dtype=np.float32)
                if action.shape != (2,):
                    action = action[:2] if len(action) > 2 else np.zeros(2)
            assert actions[i].shape == (2,), f"动作维度错误: {actions[i].shape}"
            assert self.multi_current_pos[i].shape == (2,), f"位置维度错误: {self.multi_current_pos[i].shape}"
            self.multi_current_vel[i] = actions[i] # 更新速度
            # 速度限制
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if vel_magnitude >= self.v_max:
                self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max
            # 位置更新
            self.multi_current_pos[i] += self.multi_current_vel[i] * self.time_step

        for i in range(self.num_target): # 目标
            target_idx = self.num_uav + i  
            pos = self.multi_current_pos[target_idx]
            # 直接将action作为速度，而不是加速度
            if not isinstance(actions[target_idx], np.ndarray):
                actions[target_idx] = np.array(actions[target_idx])
            
            # 目标点保持静止
            self.multi_current_vel[target_idx] = np.zeros(2)  # 初始化速度为0
            # 速度限制
            #vel_magnitude = np.linalg.norm(self.multi_current_vel[target_idx])
            #if vel_magnitude >= self.v_max_e:
            #    self.multi_current_vel[target_idx] = self.multi_current_vel[target_idx] / vel_magnitude * self.v_max_e
            # 位置更新
            self.multi_current_pos[target_idx] += self.multi_current_vel[target_idx] * self.time_step

        # Update obstacle positions
        for obs in self.obstacles:
            obs.position += obs.velocity * self.time_step
            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                if obs.position[dim] - obs.radius < 0:
                    obs.position[dim] = obs.radius
                    obs.velocity[dim] *= -1
                elif obs.position[dim] + obs.radius > self.length:
                    obs.position[dim] = self.length - obs.radius
                    obs.velocity[dim] *= -1
        # 是否已被侦查
        for uav_idx in range(self.num_uav):
            for target_idx in range(self.num_target):
                pos_uav = self.multi_current_pos[uav_idx]
                pos_target = self.multi_current_pos[self.num_uav + target_idx]
                distance = np.linalg.norm(pos_uav - pos_target)
                if distance <= self.d_capture and not self.target_detected[target_idx]:
                    self.target_detected[target_idx] = True

        # 状态更新和奖励式计算
        Collided = self.update_lasers_isCollied_wrapper() # 更新激光传感器和碰撞状态
        rewards, dones= self.cal_rewards_dones(Collided,last_d2target)   # 计算奖励和完成状态
        multi_next_obs = self.get_multi_obs() # 获取新的观察状态
        # sequence above can't be disrupted
        return multi_next_obs, rewards, dones #所有agent共享同一个终止信号

    def test_multi_obs(self):
        total_obs = []
        for i in range(self.num_uav):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0]/self.length,
                pos[1]/self.length,
                vel[0]/self.v_max,
                vel[1]/self.v_max
            ]
            total_obs.append(S_uavi)
        return total_obs

    def get_multi_obs(self):
        """生成多智能体观察状态
        UAV观察维度 = 4(自身) + 2*(num_uav-1)(其他UAV) + 3*num_target(目标信息) + 16(激光) 
                   = 4 + 2*(n-1) + 3*t + 16
        Target观察维度 = 4(自身) + 2*num_uav(UAV速度) + 16(激光) 
                     = 4 + 2n + 16
    
        参数说明：
        - 位置归一化：坐标值/环境边长 (将数值范围约束到[0,1])
        - 速度归一化：速度分量/最大速度 (UAV用v_max，目标用v_max_e)
        - 角度处理：使用sin(theta)和cos(theta)代替原始角度值（避免角度跳变问题）
        - 距离归一化：实际距离/(2√2*环境长度)（环境对角线长度的1/2）
        """
        total_obs = []
        # 处理UAV的观察空间        
        all_uav = []
        for uav_idx in range(self.num_uav):
            current_pos = self.multi_current_pos[uav_idx]
            current_vel = self.multi_current_vel[uav_idx]
        
            # 自身信息 (4维)
            self_info = [
                current_pos[0]/self.length,
                current_pos[1]/self.length,
                current_vel[0]/self.v_max,
                current_vel[1]/self.v_max
            ]
        
            # 其他UAV的相对位置 (2*(num_uav-1)维)
            other_uav_info = []
            for other_idx in range(self.num_uav):
                if other_idx != uav_idx:
                    other_pos = self.multi_current_pos[other_idx]
                    dx = (other_pos[0] - current_pos[0])/self.length
                    dy = (other_pos[1] - current_pos[1])/self.length
                    other_uav_info.extend([dx, dy])
            target_info = []
            for target_idx in range(self.num_target):
                # 计算目标绝对索引
                absolute_idx = self.num_uav + target_idx
                target_pos = self.multi_current_pos[absolute_idx]
            
                # 计算相对向量
                dx = target_pos[0] - current_pos[0]
                dy = target_pos[1] - current_pos[1]
            
                # 归一化距离（最大可能距离为环境对角线）
                distance = np.sqrt(dx**2 + dy**2)
                norm_distance = distance / (self.length * np.sqrt(2))
            
            # 计算角度并分解为sin/cos
                angle = np.arctan2(dy, dx)
                sin_theta = np.sin(angle)
                cos_theta = np.cos(angle)
            
                target_info.extend([
                    norm_distance, 
                    sin_theta,
                    cos_theta
                ])
            # 4. 激光传感器数据 (16维)
            laser_data = self.multi_current_lasers[uav_idx]
            # 合并所有观测数据
            obs = np.concatenate([
                self_info,
                other_uav_info,
                target_info,
                laser_data
            ], dtype=np.float32)
            total_obs.append(obs)
        ## 处理目标的观察空间
        for target_idx in range(self.num_target):
            absolute_idx =  self.num_uav + target_idx  # 计算绝对索引
            pos = self.multi_current_pos[absolute_idx]        
            # 1. 自身状态信息 (2维)
            # [0] x位置 / 环境边长
            # [1] y位置 / 环境边长 
            self_state = [
                pos[0] / self.length,          # 归一化x坐标
                pos[1] / self.length,          # 归一化y坐标
            ]
            # 3. 激光传感器数据 (16维)
            laser_data = self.multi_current_lasers[absolute_idx]
            # 合并所有观测数据
            obs = np.concatenate([
                self_state,
                laser_data
            ], dtype=np.float32)
            total_obs.append(obs)

        return total_obs

    def cal_rewards_dones(self,IsCollied,last_d): # 奖励函数
        rewards = np.zeros(self.num_agents)
        for take_off_idx in range(self.num_uav):
            # 没有到无人机起飞时间不算奖励
            if self.current_step < self.uav_takeoff_times[take_off_idx]:
                continue
            dones = [False] * self.num_agents
            mu1 = 0.7 # r_near 接近目标奖励系数
            self.mu1 = mu1
            mu2 = 0.4 # r_safe 安全距离奖励系数
            self.mu2 = mu2
            mu3 = 0.01 # r_multi_stage 多阶段奖励系数
            self.mu3 = mu3
            mu4 = 5 # r_finish 完成任务奖励系数
            self.mu4 = mu4
            mu5 = 2.0 # r_target 首次侦查目标奖励系数
            self.mu5 = mu5
            d_capture = 0.0001 # 捕获目标的距离, 小于这个距离视作捕获
            self.d_capture = d_capture
            d_limit = 0.75 # 目标捕获的距离限制
            mission_complete = False # 所有任务完成标志
            all_target_detected = all(self.target_detected) # 检查所有目标是否被侦查

            ## 接近目标奖励 + 重复侦查惩罚
            for i in range(self.num_uav):
                r_near = 0
                for j in range(self.num_target):
                    target_idx = self.num_uav + j
                    pos_target = self.multi_current_pos[target_idx]
                    pos = self.multi_current_pos[i] # 无人机位置
                    vel = self.multi_current_vel[i] # 无人机速度

                    dire_vec = pos_target - pos
                    d = np.linalg.norm(dire_vec)
                    v_i = np.linalg.norm(vel)
                    remaining_targets = sum(not x for x in self.target_detected)
                    discovery_bonus = 10 * (remaining_targets / self.num_target)
                    # 检查是否在捕获范围内
                    if d <= self.d_capture:
                        if not self.target_detected[j]:  # 首次发现目标
                            self.target_detected[j] = True
                            self.target_detect_count[j] += 1
                            rewards[i] += self.mu5 * discovery_bonus  # 首次侦查奖励, 与未发现数量相关

                        else:  # 重复侦查，给予惩罚
                            # 增加推离力，让UAV远离已发现的目标
                            rewards[i] += -5.0 * (1.0 / (d + 1e-5))
            
                    # 仅对未发现的目标计算接近奖励
                    if not self.target_detected[j]:
                        cos_v_d = np.dot(vel, dire_vec)/(v_i*d + 1e-3)
                        r_near += abs(2*v_i/self.v_max)*cos_v_d
            
            rewards[i] += self.mu1 * r_near / self.num_target

            ## collision reward for all UAVs:避免碰撞奖励
            if IsCollied[i]:
                rewards[i] += self.mu2 * (-20) # 如果碰撞, 给大额惩罚
            else:
                lasers = self.multi_current_lasers[i]
                r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
                rewards[i] += self.mu2 * r_safe
                # 添加位置约束奖励
            for i in range(self.num_uav):
                # 检查无人机是否超出边界
                pos = self.multi_current_pos[i]
                if pos[0] < 0 or pos[0] > self.length or pos[1] < 0 or pos[1] > self.length:
                    rewards[i] += -20.0  # 超出边界惩罚
                    # 将无人机拉回边界
                    self.multi_current_pos[i] = np.clip(pos, 0, self.length)

            ## 无人机间距奖励
            formation_valid = True # 是否有效的无人机距离
            mu6 = 0.3  # 无人机间距奖励系数
            d_min = 0.05  # 最小安全距离(相当于50米)
            d_max = 1.0   # 最大协作距离(相当于1000米)
            # 遍历所有追捕者无人机对
            for i in range(self.num_uav):  # 考虑3个无人机
                for j in range(i+1, self.num_uav):
                    pos_i = self.multi_current_pos[i] 
                    pos_j = self.multi_current_pos[j]
                    d_ij = np.linalg.norm(pos_i - pos_j)
            
                    if d_ij < d_min  or d_ij> d_max:  # 违反距离约束，给予惩罚
                        formation_valid = False
                        r_formation = -10.0
                        break
                    else:  
                        lasers_i = self.multi_current_lasers[i]
                        lasers_j = self.multi_current_lasers[j]
                        # 计算两个无人机的安全距离奖励
                        r_safe_i = (min(lasers_i) - self.L_sensor - 0.1)/self.L_sensor
                        r_safe_j = (min(lasers_j) - self.L_sensor - 0.1)/self.L_sensor
                        # 将安全距离奖励与距离适中奖励结合
                        r_formation = 0.5 + 0.3 * (r_safe_i + r_safe_j)/2

                    # 对相关的两架无人机都加上奖励
                    rewards[i] += mu6 * r_formation
                    rewards[j] += mu6 * r_formation
            # 检查所有目标都被发现
            all_targets_found = all(self.target_detected)
            # 检查是否有无人机碰撞
            no_collision = not any(IsCollied[:self.num_uav])  # 只检查UAV的碰撞状态
            # 所有条件都满足时，任务完成
            mission_complete = all_targets_found and formation_valid and no_collision
            # 更新完成状态
            dones = [mission_complete] * self.num_agents
            ## finish rewards 完成所有任务的奖励
            if mission_complete:
                rewards = [r + 100 for r in rewards]  # 给予所有agent完成奖励

            dones = [all(self.target_detected)] * self.num_agents # 所有agent共享同一个终止信号

        return rewards,dones

    def update_lasers_isCollied_wrapper(self):
        self.multi_current_lasers = []
        dones = []
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []
            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos,obs_pos,r,self.L_sensor,self.num_lasers,self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)
            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)
            self.multi_current_lasers.append(current_lasers)
            dones.append(done)
        return dones

    def render(self):

        plt.clf() # 清楚当前图像
        
        # load UAV icon
        uav_icon = mpimg.imread('UAV.png')
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for i in range(self.num_uav):
            # 获取当前位置和速度
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            # 记录历史轨迹
            self.history_positions[i].append(pos)
            trajectory = np.array(self.history_positions[i])
            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)
            # 计算无人机朝向角度
            angle = np.arctan2(vel[1], vel[0])

            # plt.scatter(pos[0], pos[1], c='b', label='hunter')
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            # plt.imshow(uav_icon, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            # plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(pos[0] - 0.05, pos[0] + 0.05, pos[1] - 0.05, pos[1] + 0.05))
            icon_size = 0.01  # 设置图标大小
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

            # # Visualize laser rays for each UAV(can be closed when unneeded)
            # lasers = self.multi_current_lasers[i]
            # angles = np.linspace(0, 2 * np.pi, len(lasers), endpoint=False)
            
            # for angle, laser_length in zip(angles, lasers):
            #     laser_end = np.array(pos) + np.array([laser_length * np.cos(angle), laser_length * np.sin(angle)])
            #     plt.plot([pos[0], laser_end[0]], [pos[1], laser_end[1]], 'b-', alpha=0.2)

        # plot target
        #plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        #self.history_positions[-1].append(copy.deepcopy(self.multi_current_pos[-1]))
        #trajectory = np.array(self.history_positions[-1])
        #plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)
        # 绘制目标点
        for i in range(self.num_target):
            target_idx = self.num_uav + i
            pos = self.multi_current_pos[target_idx]
        
        # 记录目标轨迹
        self.history_positions[target_idx].append(copy.deepcopy(pos))
        trajectory = np.array(self.history_positions[target_idx])
        
        # 绘制目标轨迹和点（使用不同颜色区分不同目标）
        colors = ['r', 'm', 'y', 'g', 'c']  # 为不同目标设置不同颜色
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                f'{colors[i]}-', alpha=0.3, linewidth=1)
        
        # 目标点标记
        if self.target_detected[i]:  # 如果目标已被发现
            marker_style = '*'  # 使用星形标记
            marker_size = 150
        else:  # 未被发现的目标
            marker_style = 'o'  # 使用圆形标记
            marker_size = 100

        # 绘制目标点（红色星形）
        plt.scatter(pos[0], pos[1], 
                   c=colors[i], marker=marker_style, 
                   s=marker_size, label=f'Target_{i}')

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.length+0.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # 图例
        plt.draw()
        # plt.pause(0.01)
        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        
        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime(self, frame_num):
        plt.clf()
        
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_uav):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]
            angle = np.arctan2(vel[1], vel[0])
            self.history_positions[i].append(pos)
            
            trajectory = np.array(self.history_positions[i])
            for j in range(len(trajectory) - 1):
                color = cm.viridis(j / len(trajectory))  # 使用 viridis colormap
                plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], color=color, alpha=0.7)
            # plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=1)

            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.01
            plt.imshow(uav_icon, transform=t + plt.gca().transData, extent=(-icon_size/2, icon_size/2, -icon_size/2, icon_size/2))

        # 2. 绘制所有目标点
        colors = ['r', 'm', 'y', 'g', 'c']
        for i in range(self.num_target):
            target_idx = self.num_uav + i
            pos = self.multi_current_pos[target_idx]
        
            self.history_positions[target_idx].append(copy.deepcopy(pos))
            trajectory = np.array(self.history_positions[target_idx])
        
            plt.plot(trajectory[:, 0], trajectory[:, 1], 
                    f'{colors[i]}-', alpha=0.3)
            plt.scatter(pos[0], pos[1], 
                   c=colors[i], 
                   marker='*' if self.target_detected[i] else 'o',
                   s=100, label=f'Target_{i}')

        #plt.scatter(self.multi_current_pos[-1][0], self.multi_current_pos[-1][1], c='r', label='Target')
        #pos_e = copy.deepcopy(self.multi_current_pos[-1])
        #self.history_positions[-1].append(pos_e)
        #trajectory = np.array(self.history_positions[-1])
        #plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.3)

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.position, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.legend(loc='upper right')
        plt.draw()

    def close(self):
        plt.close()

class obstacle():
    def __init__(self, length=2):
        self.position = np.array([0.0, 2.0]) # 障碍物位置
        # self.position = np.random.uniform(low=0.45, high=length-0.55, size=(2,)) # 障碍物位置随机
        angle = np.random.uniform(0, 2 * np.pi)
        # speed = 0.03 
        speed = 0.00 # to make obstacle fixed
        self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)]) # 障碍物的速度
        self.radius = 0 # 障碍物的半径, 第一小题设置为0