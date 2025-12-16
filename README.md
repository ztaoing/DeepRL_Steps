## 强化学习-原理

1. BASIC            【鱼书】深度学习入门-强化学习 [课件及笔记](./BASIC)
2. DRL              【王树森】深度强化学习 [课件及笔记](./DRL)
3. OPEN AI 强化学习手册 [官网地址](https://spinningup.openai.com/en/latest/index.html)
4. 李宏毅-强化学习-PPO 【[视频地址](https://www.bilibili.com/video/BV18r421j7S4?spm_id_from=333.788.videopod.episodes&vd_source=f397e73b314ac775b2d6145b41327fa0) 】
5. 李宏毅-强化学习-2025【[视频地址](https://www.bilibili.com/video/BV15hw9euExZ/?spm_id_from=333.337.search-card.all.click&vd_source=f397e73b314ac775b2d6145b41327fa0)】
6. easy-rl   [在线地址](https://datawhalechina.github.io/easy-rl/#/)
7. Mathematical-RL  [【赵世钰】强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS/?）

## 强化学习-代码实现

* Hands-on-RL       [【愈勇等】动手学强化学习 ](./Hands-on-RL)
* cleanrl         [原地址](https://github.com/vwxyzjn/cleanrl)
* joyrl  [入门强化学习的代码生态](https://datawhalechina.github.io/joyrl-book/#/)
* easy-rl   [在线地址](https://datawhalechina.github.io/easy-rl/#/)
* notes-on-reinforcement-learning   [在线阅读地址](https://newfacade.github.io/notes-on-reinforcement-learning/01-intro.html#)
* 强化学习算法实现 [DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch)
* [复现deepseek-r1](https://github.com/FareedKhan-dev/train-deepseek-r1?tab=readme-ov-file#grpo-training-loop)
## 强化学习算法 github地址
* [DAPO-github](https://github.com/BytedTsinghua-SIA/DAPO)
* [ASPO-github](https://github.com/wizard-III/Archer2.0)

## 强化学习框架
* trl  [trl](https://github.com/willccbb/trl)
* veRL [veRL](https://github.com/volcengine/verl)
## RLHF 

* deepspeed-chat [deepspeed-chat](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-chat)

## 强化学习论文

### PPO
* [PPO](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/PPO%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96.pdf)

* [[复旦]PPO-Max:Secrets of RLHF in Large Language Models Part I- PPO](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/1%E3%80%81PPO-Max%20Secrets%20of%20RLHF%20in%20Large%20Language%20Models%20Part%20I-%20PPO.pdf)
* [[复旦]PPO-Max:github地址](https://github.com/OpenLMLab/MOSS-RLHF)
### DeepSeek及clip
* [DeepSeek-GRPO]()
* [DeepSeek-R1](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/DeepSeek-R1-%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.pdf)

* [[推荐]重新思考下 PPO-Clip](https://zhuanlan.zhihu.com/p/1950985242098799047)
* [DAPO-字节](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E5%AD%97%E8%8A%82clip-higher%EF%BC%9ADAPO-%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.pdf)
* [ASPO-非对称重要性采样策略优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/ASPO-%20Asymmetric%20Importance%20Sampling%20Policy%20Optimization.pdf)
* [Soft Clip 机制：CISPO-MiniMax-M1- Scaling Test-Time Compute Efficiently with Lightning Attention](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Soft%20Clip%20%E6%9C%BA%E5%88%B6%EF%BC%9ACISPO-MiniMax-M1-%20Scaling%20Test-Time%20Compute%20Efficiently%20with%20Lightning%20Attention.pdf)

* [DPO-直接偏好优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Direct%20Preference%20Optimization-%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.pdf)
* [MAPO-混合优势策略优化]()

### GRPO在机器领域的应用
* [Extending Group Relative Policy Optimization to ContinuousControl: A Theoretical Framework for Robotic Reinforcement Learning]()
### KL散度
* [[推荐]为何在线强化学习能有效缓解灾难性遗忘？Why Online Reinforcement Learning Forgets Less](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E4%B8%BA%E4%BD%95%E5%9C%A8%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%83%BD%E6%9C%89%E6%95%88%E7%BC%93%E8%A7%A3%E7%81%BE%E9%9A%BE%E6%80%A7%E9%81%97%E5%BF%98%EF%BC%9FWhy%20Online%20Reinforcement%20Learning%20Forgets%20Less.pdf)
* [[for LLM]RL当你的 KL 散度正则化在“裸奔”On a few pitfalls in KL divergence gradient estimation for ]()
### 熵
* [[熵]1-关注的是宏观的、全局的“策略熵”The Entropy Mechanism of Reinforcement Learning for Reasoning Language Model](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/1-%E5%85%B3%E6%B3%A8%E7%9A%84%E6%98%AF%E5%AE%8F%E8%A7%82%E7%9A%84%E3%80%81%E5%85%A8%E5%B1%80%E7%9A%84%E2%80%9C%E7%AD%96%E7%95%A5%E7%86%B5%E2%80%9DThe%20Entropy%20Mechanism%20of%20Reinforcement%20Learning%20for%20Reasoning%20Language%20Model.pdf)
* [[熵]2-关注的是微观的、局部的“Token 级熵”Beyond the 80,20 Rule- High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/2-%E5%85%B3%E6%B3%A8%E7%9A%84%E6%98%AF%E5%BE%AE%E8%A7%82%E7%9A%84%E3%80%81%E5%B1%80%E9%83%A8%E7%9A%84%E2%80%9CToken%20%E7%BA%A7%E7%86%B5%E2%80%9DBeyond%20the%2080%2C20%20Rule-%20High-Entropy%20Minority%20Tokens%20Drive%20Effective%20Reinforcement%20Learning%20for%20LLM%20Reasoning.pdf)

*  [动态平衡探索与利用（在难题上探索、在简单题上利用）-DACE]()
*  [利用不确定性面向长序列LLM智能体的熵调制策略梯度Harnessing Uncertainty  Entropy Modulated Policy Gradients for Long-Horizon LLM Agents](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E5%88%A9%E7%94%A8%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E9%9D%A2%E5%90%91%E9%95%BF%E5%BA%8F%E5%88%97LLM%E6%99%BA%E8%83%BD%E4%BD%93%E7%9A%84%E7%86%B5%E8%B0%83%E5%88%B6%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6Harnessing%20Uncertainty%20%20Entropy%20Modulated%20Policy%20Gradients%20for%20Long-Horizon%20LLM%20Agents.pdf)
*  [探索不应是盲目和全局的，而应是有选择性的Rethinking Entropy Regularization in Large Reasoning model]()

### 是否一定要控制熵
* [deepseek-math-V1]()


### 奖励模型的设计
* [通用奖励模型的推理时扩展Inference-Time Scaling for Generalist Reward Modeling](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E8%AE%A9AI%E5%AD%A6%E4%BC%9A%E8%87%AA%E6%88%91%E6%89%B9%E8%AF%84-%E9%80%9A%E7%94%A8%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86%E6%97%B6%E6%89%A9%E5%B1%95Inference-Time%20Scaling%20for%20Generalist%20Reward%20Modeling.pdf)
### 阿里：“目标”（序列级）和“手段”（token级）之间的不匹配
* [一阶近似Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%5B%E9%98%BF%E9%87%8C%5DStabilizing%20Reinforcement%20Learning%20with%20LLMs-%20Formulation%20and%20Practices.pdf)
* [【对整个 response 进行裁剪】GSPO-组序列策略优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/GSPO%EF%BC%9AGroup%20Sequence%20Policy%20Optimization.pdf)

## 神经网络
* http://udlbook.github.io/udlbook  深度学习中的算法背后的原理
* https://github.com/changyeyu/LLM-RL-Visualized  图解大模型算法
* https://www.rethink.fun  大模型核心技术和应用

## 分布式
* Ray-利用Ray进行大模型的数据处理、训练、推理和部署 [Ray rllib github地址](https://github.com/ray-project/ray/tree/master/rllib)
