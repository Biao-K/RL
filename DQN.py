import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 200   # target update frequency(频率)
MEMORY_CAPACITY = 4000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  #2个动作
N_STATES = env.observation_space.shape[0]  #4个状态值 位置 移动速度  角度  移动角度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # 用于目标更新  学习的步数
        self.memory_counter = 0                                         # 用于存储内存
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) #采用Adam优化器
        self.loss_func = nn.MSELoss()                                     #采用均方误差
    def choose_action(self, x):  #x为当前状态的4个值
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  #在数据的第0维处增加一维
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy #贪婪取法
            actions_value = self.eval_net.forward(x)  ##传入eval_net获取下一个的动作
            action = torch.max(actions_value, 1)[1].data.numpy()  ##返回这一行中最大值的索引
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            # action = random.sample(N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    def store_transition(self, s, a, r, s_):  #s和s_都为4个值，分别为  位置 移动速度  角度  移动角度
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory #更新经验
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition  #将第index经验替换为transition
        self.memory_counter += 1
    def learn(self):
        # target parameter update 目标参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  ## 每学习200步将eval_net的参数赋值给target_net
        self.learn_step_counter += 1
        # sample batch transitions  #选取过渡
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) #从MEMORY_CAPACITY随机选取BATCH_SIZE个
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  #第一个状态
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)) #动作
        print("--------")
        print(b_a)
        print("-----")
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]) #得分
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]) #下一个状态
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)   # shape (batch, 1) 当前状态的Q值使用eval_net计算
        # print("++++++")
        # print(self.eval_net(b_s))
        # print(self.eval_net(b_s).gather(1,b_a))
        # print("+++++++")

        q_next = self.target_net(b_s_).detach()   #使用target_net计算下一步Q值  # detach from graph, don't backpropagate detach防止targent——net反向传播
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()   #zer——grad设置所有优化器的梯度为0
        loss.backward()   #反向传播
        self.optimizer.step()    #执行下个优化
dqn = DQN()
pl = []
print('\nCollecting experience...')
for i_episode in range(2000):
    s = env.reset()  #初始化环境 s为环境的4个值位置 移动速度  角度  移动角度
    # print("*******")
    # print(s)
    # print("*******")
    ep_r = 0
    while True:
        # env.render()  ##渲染环境
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)  #观察，奖励，完成，信息 eg:[ 0.00147281  0.21595768  0.05230965 -0.19926615] 1.0 False {}
        # print("------")
        # print(s_, r, done, info)
        # print("------")
        # modify the reward
        x, x_dot, theta, theta_dot = s_   #位置 移动速度  角度  移动角度
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  #env.x_threshold = 2.4
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  #env.theta_threshold_radians = 0.209
        r = r1 + r2
        # print("----")
        # print(r1,r2,r)
        # print("-----")
        dqn.store_transition(s, a, r, s_)  #保存在经验池中
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:  ##当经验池满了时进行学习
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                pl.append(round(ep_r,2))
        if done:
            break
        s = s_

plt.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.figure(figsize=(10,8))
plt.title("模型得分",size = 20)
plt.plot(range(len(pl)),pl)
plt.xticks(fontsize = 20) #设置坐标轴数字大小
plt.yticks(fontsize = 20)
plt.xlabel("次数",size = 20)
plt.ylabel("得分",size = 20)
plt.show()