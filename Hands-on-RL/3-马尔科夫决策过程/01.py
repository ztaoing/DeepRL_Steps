import numpy as np

np.random.seed(0)
# 定义状态转移概率矩阵
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)
# 定义奖励函数
rewards = [-1, -2, -2, 10, 1, 0]
# 定义这个因子
gamma = 0.5


# 给定一条序列，计算从某个索引（起始状态）开始--》序列最后（终止状态） 得到的回报
def compute_return(start_index, chain, gamma):
    G = 0

    for i in range(start_index, len(chain)):
        G = gamma * G + rewards[chain[i] - 1]
        print(i)
        print("--")
    return G


# 状态序列： s1 -> s2 -> s3 -> s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index=start_index, chain=chain, gamma=gamma)
print("根据本序列计算得到的回报为：%s" % G)
