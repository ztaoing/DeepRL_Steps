import numpy as np
import matplotlib.pyplot as plt

theta = np.array([8, 9, 7, 8.5, 9.5, 10, 6, 7, 7.5, 8, 8.5, 9, 9.2, 8.8])
days = np.arange(len(theta))
print(days)

# 设置beta值，选择一个‘稳重’的调配室
beta = 0.9

# 初始化
v = 0  # 起始的甜度
v_standard = []
v_corrected = []

for t in range(len(theta)):
    # 标准的EWA
    v = beta * v + (1 - beta) * theta[t]
    v_standard.append(v)

    # 带偏差修正的EWA
    v_corr = v / (1 - beta ** (t + 1))
    v_corrected.append(v_corr)
