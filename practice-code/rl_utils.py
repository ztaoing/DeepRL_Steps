from tqdm import tqdm
import numpy as np
import torch
import colletions
import random

# 经验回放池
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitons = random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done = zip(*transitons)
        return np.array(state),action,reward,np.array(next_state),done
    def size(self):
        return len(self.buffer)
    
def moving_average(a,window_size):
    cumulative_sum =np.cumsum(np.insert(a,0,0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size])/window_size
    r = np.arange(1,window_size-1,2)
    
    begin = np.cumsum(a[:window_size-1])[::2]/r
    end = (np.cumsum(a[:-windos_size:-1])[::2]/r)[::-1]
    
    return np.oncatenate((begin,middle,end))

# 在线模式
def train_on_policy_agent(env,agent,num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10),desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,_=env.step(action)
                    #放入到transition字典中
                    transition_dict['states'].append(sates)
                    transition_dict['actions'].append(ation)
                    transition_dict['next_state'].append(next_state)
                    transition_dict['reward'].append(reward)
                    transition_dict['dones'].append(done)
                    #更新
                    state = next_state
                    episode_return +=reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0 :# 每10次输出tqdm
                    pbar.set_postfix({'episode':'%d' % (num_episodes/10 * i + i_episode+1), 'return':'%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
    return return_list


# 离线模式
def train_off_policy_agent(env,agent,num_episodes,replay_buffer,minimal_size,batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10),desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = agent.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done,_=env.step(action)
                    
                    # 将采样数据加入到repaly_buffer中
                    replay_buffer.add(state,action,reward,next_state,done)
                    
                    # 更新
                    state = next_state
                    episode_return += reward
                    # 到达一定容量后才能使用这些经验数据
                    if replay_buffer.size() > minimal_size:
                        b_s,b_a,b_r,b_ns,b_d = replay_buffer.sample(batch_size)
                        # 放入字典中
                        transition_dict = {'states':b_s,'actions':b_a,'next_states':b_ns,'rewards':b_r,'dones':b_d}
                        # 更新
                        agent_update(transition_dict)
                    return_list.append(episode_return)
                    # tqdm
                    if (i_episode + 1) % 10 == 0 :# 每10次输出tqdm
                    pbar.set_postfix({'episode':'%d' % (num_episodes/10 * i + i_episode+1), 'return':'%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
    return return_list
                    
# 计算优势
def compute_advantage(gamma,lmbda,td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list,dtype = torch.float)