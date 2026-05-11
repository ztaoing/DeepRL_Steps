
# 观察结果 ≠ 客观事实，而是「观察方式」的产物。
## 强化学习-原理

1. BASIC            【鱼书】深度学习入门-强化学习 [课件及笔记](./BASIC)
2. DRL              【王树森】深度强化学习 [课件及笔记](./DRL)
3. Hands-on-RL       [【愈勇等】动手学强化学习++ ](./Hands-on-RL)
4. OPEN AI 强化学习手册 [官网地址](https://spinningup.openai.com/en/latest/index.html)
5. 李宏毅-强化学习-PPO 【[视频地址](https://www.bilibili.com/video/BV18r421j7S4?spm_id_from=333.788.videopod.episodes&vd_source=f397e73b314ac775b2d6145b41327fa0) 】
6. 李宏毅-强化学习-2025【[视频地址](https://www.bilibili.com/video/BV15hw9euExZ/?spm_id_from=333.337.search-card.all.click&vd_source=f397e73b314ac775b2d6145b41327fa0)】
7. [人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)
8. easy-rl   [在线地址](https://datawhalechina.github.io/easy-rl/#/)
9. Mathematical-RL  [【赵世钰】强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS/?）
10. [RLHF-huggingface](https://huggingface.co/blog/zh/rlhf)
## 强化学习-代码实现


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


![论文关联1](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/rl_ai.png?v=9)
# 强化学习论文
* [TODOStatistical Reinforcement Learning in the Real World: A Survey of  Challenges and Future Directions]()

  全面综述：RL在现实世界落地的未来方向

##  RL(pass@1)与最大似然估计MLE(pass@k)
* [Maximum Likelihood Reinforcement Learning]()

  MaxRL建立起RL(pass@1)与最大似然估计MLE(pass@k)之间的桥梁


## 技巧还是陷阱？
* [JustRL: Scaling a 1.5B LLM with a Simple RL Recipe]()

  当我们试着加入一些"应该有用"的优化时，性能反而下降了
  
* [Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning]()

  技巧还是陷阱？从bese模型和aligned模型的角度观察

![RL-tools](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/RL-tool.png?v=1)

![RL-tools](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/advantage.png?v=1)
## PPO
* [PPO（通过裁剪重要性权重，实现稳定策略更新）](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/PPO%E8%BF%91%E7%AB%AF%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96.pdf)
    - <b>token-level的advantage</b>

* [[复旦]PPO-Max:Secrets of RLHF in Large Language Models Part I- PPO](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/1%E3%80%81PPO-Max%20Secrets%20of%20RLHF%20in%20Large%20Language%20Models%20Part%20I-%20PPO.pdf) [[复旦]PPO-Max:github地址](https://github.com/OpenLMLab/MOSS-RLHF)
* [[分析了普通 IS 与加权 IS（WIS）在 off-policy TD 中的方差特性，证明 WIS 更稳定]Weighted importance sampling for off-policy learningwith linear function approximation]()



## DeepSeek及clip
* [DeepSeek-GRPO]()
    - <b>sequence-level的advantage</b>
* [DeepSeek-R1](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/DeepSeek-R1-%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.pdf)

* [[推荐]重新思考下 PPO-Clip](https://zhuanlan.zhihu.com/p/1950985242098799047)
* [DAPO-字节](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E5%AD%97%E8%8A%82clip-higher%EF%BC%9ADAPO-%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.pdf)
* [ASPO-非对称重要性采样策略优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/ASPO-%20Asymmetric%20Importance%20Sampling%20Policy%20Optimization.pdf)
* [Soft Clip 机制：CISPO-MiniMax-M1- Scaling Test-Time Compute Efficiently with Lightning Attention](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Soft%20Clip%20%E6%9C%BA%E5%88%B6%EF%BC%9ACISPO-MiniMax-M1-%20Scaling%20Test-Time%20Compute%20Efficiently%20with%20Lightning%20Attention.pdf)

* [DPO-直接偏好优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Direct%20Preference%20Optimization-%20Your%20Language%20Model%20is%20Secretly%20a%20Reward%20Model.pdf)
* [Understanding R1-Zero-Like Training: A Critical Perspective](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Understanding%20R1-Zero-Like%20Training%20A%20Critical%20Perspective.pdf)
* [FIPO（Future-KL Influenced Policy Optimization）]()
    - FIPO 追踪的是每个Token引发的概率偏移(实时追踪每一个Token对后续推理轨迹的实际影响)





## GRPO在机器领域的应用（TODO）
* [Extending Group Relative Policy Optimization to ContinuousControl: A Theoretical Framework for Robotic Reinforcement Learning]()


## (显示/隐式)KL散度
* [k3估计器：Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)
* [[推荐]Why Online Reinforcement Learning Forgets Less](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E4%B8%BA%E4%BD%95%E5%9C%A8%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%83%BD%E6%9C%89%E6%95%88%E7%BC%93%E8%A7%A3%E7%81%BE%E9%9A%BE%E6%80%A7%E9%81%97%E5%BF%98%EF%BC%9FWhy%20Online%20Reinforcement%20Learning%20Forgets%20Less.pdf)

  为何在线强化学习能有效缓解灾难性遗忘？
  
  [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting]()
  - 支持“on-policy SFT 可减轻遗忘”的观点
* [[for LLM]On a few pitfalls in KL divergence gradient estimation for ](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/On%20a%20few%20pitfalls%20in%20KL%20divergence%20gradient%20estimation%20for%20RL%E5%BD%93%E4%BD%A0%E7%9A%84%20KL%20%E6%95%A3%E5%BA%A6%E6%AD%A3%E5%88%99%E5%8C%96%E5%9C%A8%E2%80%9C%E8%A3%B8%E5%A5%94%E2%80%9D.pdf)

  RL当你的 KL 散度正则化在“裸奔”
  
* [A Comedy of Estimators On KL Regularization in RL Training of LLMs](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/A%20Comedy%20of%20Estimators%20On%20KL%20Regularization%20in%20RL%20Training%20of%20LLMs.pdf)

  分析了两种主流 KL 估计器（K1 和 K3）在两种放置位置（Reward 和 Loss）下的梯度特性

## 熵（探索）
* [[熵]1-The Entropy Mechanism of Reinforcement Learning for Reasoning Language Model](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/1-%E5%85%B3%E6%B3%A8%E7%9A%84%E6%98%AF%E5%AE%8F%E8%A7%82%E7%9A%84%E3%80%81%E5%85%A8%E5%B1%80%E7%9A%84%E2%80%9C%E7%AD%96%E7%95%A5%E7%86%B5%E2%80%9DThe%20Entropy%20Mechanism%20of%20Reinforcement%20Learning%20for%20Reasoning%20Language%20Model.pdf)
  
  关注的是宏观的、全局的“策略熵”
  
* [[熵]2-Beyond the 80,20 Rule- High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/2-%E5%85%B3%E6%B3%A8%E7%9A%84%E6%98%AF%E5%BE%AE%E8%A7%82%E7%9A%84%E3%80%81%E5%B1%80%E9%83%A8%E7%9A%84%E2%80%9CToken%20%E7%BA%A7%E7%86%B5%E2%80%9DBeyond%20the%2080%2C20%20Rule-%20High-Entropy%20Minority%20Tokens%20Drive%20Effective%20Reinforcement%20Learning%20for%20LLM%20Reasoning.pdf)
 
  - 关注的是微观的、局部的“Token 级熵”
*  [Harnessing Uncertainty  Entropy Modulated Policy Gradients for Long-Horizon LLM Agents](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E5%88%A9%E7%94%A8%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E9%9D%A2%E5%90%91%E9%95%BF%E5%BA%8F%E5%88%97LLM%E6%99%BA%E8%83%BD%E4%BD%93%E7%9A%84%E7%86%B5%E8%B0%83%E5%88%B6%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6Harnessing%20Uncertainty%20%20Entropy%20Modulated%20Policy%20Gradients%20for%20Long-Horizon%20LLM%20Agents.pdf)

   - 利用不确定性面向长序列LLM智能体的熵调制策略梯度
  
*  [TODO Rethinking Entropy Regularization in Large Reasoning model]()

   - 探索不应是盲目和全局的，而应是有选择性的
   
## 是否一定要控制熵
* [deepseek-math-V1]()
  
## 稀疏性

* [TODO-Reinforcement Learning Finetunes Small Subnetworks in Large Language Models]()
    - RL引起的参数更新稀疏性
    - 这种稀疏性主要源于强化学习微调的数据特性

* [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models]()

    指出 RL 微调是“局部更新”，而非全局重塑，因此更容易被后续训练干扰
    - 实际上，它只改动了模型 5%-30% 的权重，剩下的部分几乎纹丝不动。这和 SFT（监督微调）那种“地毯式轰炸”的更新模式完全不同。

    
* [Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Sparse%20but%20Critical%20A%20Token-Level%20Analysis%20of%20Distributional%20Shifts%20in%20RLVR%20Fine-Tuning%20of%20LLMs.pdf)
    - 在绝大多数生成步骤中，强化学习模型与基础模型的预测分布近乎一致，仅在少部分特定位置出现明显的发散。
    - 少部分词元决定整体性能
    - 在出现较大分布变化的词元位置，RLVR 主要是在基础模型原有的候选词元集合内进行概率重分配和排序调整，较少提升原本在基础模型中处于低概率尾部的词元。
    - DAPO 算法能够同时修改初始高熵和低熵的预测，这体现了其覆盖甚至推翻基础模型自信预测的能力。
    - SimpleRL 则倾向于在基础模型具有较高熵的区域集中产生散度，反映出一种更为保守的更新策略
    - base model 引入的稀疏token，改变了推理的轨迹
   ![RL-tools](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/sparseRL.png?v=1) 
  





## 从幅度到方向
### reasoning direction: pointing from base to RLVR distribution
* [On the Direction of RLVR Updates for LLM Reasoning: Identification and Exploitation]()
    - 大小度量(熵和KL散度)的分布直方图在Base和RLVR模型之间几乎一模一样
    - <b>稀疏性源于RLVR对低概率Token的天然聚焦</b>
    - 测试时外推是在<b>训练完成后</b>放大RLVR学到的信号，
    - 而训练时重加权则是在<b>训练过程中</b>主动强化这些信号
    - RLVR 的策略梯度稀疏地集中在低概率token上
### Low-Probability Tokens
* [Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs]()
    1. 优势重加权（Advantage Reweighting, AR）：通过重新调整不同概率词元的优势（advantage）权重，直接削弱低概率词元的影响力。
    2. 低概率词元隔离（Low-Probability Token Isolation, Lopti）：将更新过程分解为两个阶段，先更新低概率词元，再更新高概率词元，通过隔离来避免梯度干扰。
    
        <b>既然高概率词元的梯度那么小，我们干脆在更新时忽略它们，只用中低概率的词元不就行了吗？</b>
         - 图6(a)的实验否定了这一想法。结果显示，如果屏蔽掉高概率词元，模型的性能会比基线 GRPO 更差。这说明高概率词元虽然梯度信号微弱，但它们对模型的贡献是不可或缺的。
        <b>Lopti 的更新顺序是成功的关键：</b>
         - Lopti 的核心是“先低后高”的更新顺序。如果把顺序颠倒，变成“先高后低”，会发生什么？图6(b)给出了答案——训练过程在第四个 epoch 后彻底崩溃，性能远差于基线。
         - <b>只有先处理高梯度、影响大的低概率词元，才能为后续高概率词元的精细调整创造条件。</b>

## 优势归一化：
* [group-level:GRPO]()
* [batch-level:REINFORCE++]()
* [MaxRL：使用成功样本的数量进行归一化，而不是总样本数量]()

## 损失计算：response-level还是token-level
* [response-level:GRPO]()
* [token-level:DAPO](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E5%AD%97%E8%8A%82clip-higher%EF%BC%9ADAPO-%20An%20Open-Source%20LLM%20Reinforcement%20Learning%20System%20at%20Scale.pdf)


## 幻觉
* [Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Mitigating%20LLM%20Hallucination%20via%20Behaviorally%20Calibrated%20Reinforcement%20Learning.pdf)
  
  用行为校准 RL 抑制模型幻觉
  
* [Why Language Models Hallucinate]()
  
  为什么大模型出现幻觉？

## 奖励的稀疏性和二元性(对/错)

## 奖励模型的设计
* [通用奖励模型的推理时扩展Inference-Time Scaling for Generalist Reward Modeling](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E8%AE%A9AI%E5%AD%A6%E4%BC%9A%E8%87%AA%E6%88%91%E6%89%B9%E8%AF%84-%E9%80%9A%E7%94%A8%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86%E6%97%B6%E6%89%A9%E5%B1%95Inference-Time%20Scaling%20for%20Generalist%20Reward%20Modeling.pdf)

## 单奖励与多奖励
* [GRPO](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/DeepSeek-R1-%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.pdf)
  
  单奖励
  
* [GDPO: Group reward-Decoupled Normalization  Policy Optimization for Multi-reward RL  Optimization](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/GDPO%20Group%20reward-Decoupled%20Normalization%20Policy%20Optimization%20for%20Multi-reward%20RL%20Optimization.pdf)
  
  多奖励+先归一化后加

## 阿里：“目标”（序列级）和“手段”（token级）之间的不匹配
* [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%5B%E9%98%BF%E9%87%8C%5DStabilizing%20Reinforcement%20Learning%20with%20LLMs-%20Formulation%20and%20Practices.pdf)
  
  一阶近似
  
* [GSPO-组序列策略优化](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/GSPO%EF%BC%9AGroup%20Sequence%20Policy%20Optimization.pdf)
  
  【对整个 response 进行裁剪】

## RL是否真能超越base model？
* [ Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/Does%20Reinforcement%20Learning%20Really%20Incentivize%20Reasoning%20Capacity%20in%20LLMs%20Beyond%20the%20Base%20Model.pdf)

  强化学习是否真的在Llms中激发了超出基础模型的推理能力?

* [On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/On%20the%20Interplay%20of%20Pre-Training%2C%20Mid-Training%2C%20and%20RL%20on%20Reasoning%20Language%20Models.pdf)

  预训练（Pre-Training）、中期训练（Mid-Training）和基于强化学习的后训练（RL Post-Training）
## RL 能让模型学会全新的推理模式，实现 "能力扩展"
* [RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?]()

## 从离散的token序列转向连续的注意力分布
* [Reinforced Attention Learning (RAL)]()


## mid-training 中训练：领域指数的重要性
* [OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling]()

  爬坡似的接触特定的领域数据：基础领域知识-》专业领域知识

## RL 与 SFT
* [[推荐]Why Online Reinforcement Learning Forgets Less](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/%E4%B8%BA%E4%BD%95%E5%9C%A8%E7%BA%BF%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%83%BD%E6%9C%89%E6%95%88%E7%BC%93%E8%A7%A3%E7%81%BE%E9%9A%BE%E6%80%A7%E9%81%97%E5%BF%98%EF%BC%9FWhy%20Online%20Reinforcement%20Learning%20Forgets%20Less.pdf)

  为何在线强化学习能有效缓解灾难性遗忘？
  
* [The Path Not Taken: RLVR Provably Learns Off the Principals](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/The%20Path%20Not%20Taken%20RLVR%20Provably%20Learns%20Off%20the%20Principals.pdf)

  RLVR微调的本质是“非主成分学习”！SFT微调的是“主成分”
  
* [TODO主权重：LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning]()


## RL训练(Scaling Laws)
* [The Art of Scaling Reinforcement Learning Compute for LLMs](https://github.com/ztaoing/DeepRL_Steps/blob/main/arxiv/The%20Art%20of%20Scaling%20Reinforcement%20Learning%20Compute%20for%20LLMs.pdf)

  RL 的 scaling 到底有没有规律可循？
  
* [工具综合推理：ToolRL: Reward is All Tool Learning Needs]()




## 多模态
### ：“对于视觉理解任务”，从优化结果（token），转向优化过程（内部信息分配）

对于视觉理解任务，显式的语言逻辑（Verbalized Logic）可能并不是必须的。

![token_to_process](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/token_to_process.png?v=2)

* [Reinforced Attention Learning]()

  - MLLM（多模态大模型）中从Next-token Prediction（下一个词预测）-->Attention Distribution(注意力分布)
  - 为什么 LLM 的 CoT 经验在多模态感知任务上失效了？
  - 作者的目标是：让高奖励的回复对应的注意力模式被保留和增强
  - 去 CoT 化
  - 强化学习不应该只停留在输出层。Transformer 内部丰富的中间状态（Attention, Activations）其实蕴含着巨大的可优化空间



## 稀疏奖励  

![token_to_process](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/erl.png?v=1)

### 经验强化学习（Experiential Reinforcement Learning, ERL）
   学习者在观察到结果后，会反思发生了什么，形成修正后的内部模型，并在后续的尝试中应用这些修正
* [Experiential Reinforcement Learning, ERL]()
  - 传统的强化学习通常将复杂的环境反馈压缩成一个简单的标量优化信号（scalar optimization signals），这要求策略在没有明确方向的探索中隐式地发现纠正结构
  - 引入经验学习机制，模型可以像人一样，把反馈转化为具体的中间推理（即反思过程），从而进行显式的纠正，大大提高了学习效率和针对性
  - 经验学习：环境的反馈 --> 一段具体的反思
  - 将反思视为中间推理信号
  - 让它先尝试，拿到反馈后进行文字反思，然后再做一次修正尝试，最后将成功的经验“内化”到基础策略中
  - 【反常情况】在 Olmo3-7B-Instruct 挑战 Sokoban 的设定中，无记忆变体反而略微超越了完整的 ERL。


## 蒸馏
### On-Policy Distillation
* [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
  - 训练与测试的统一：学生写，老师改
  - SFT是一种破坏性的训练（哪怕目标分布与原始分布完全一致）
* [Reinforced Attention Learning]()

  使用On-Policy Distillation方案
* [蒸馏实现](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)

### 自蒸馏
* [Experiential Reinforcement Learning, ERL]()

  使用自蒸馏方案
  
![On-Policy-Distillation](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/On-Policy-Distillation.png?v=2)

* [Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?]()
    研究表明，尽管自我蒸馏在化学问答、代码生成等领域能够缩短推理路径并提升模型性能，但在数学推理领域，该方法会导致模型性能出现较大幅度的下降。
  - 在化学领域，SDPO 在缩短回复长度的同时快速提升了分数
  - 而在数学领域（DAPO-Math-17k 数据集），SDPO 随着训练步数的增加，其评估分数却低于持续增长的 GRPO。
  - “认知不确定性表达”（Epistemic Verbalization）
  - 语言模型输出中的认知不确定性表达是支持其执行纠错和寻找解答的核心组成部分。
## Awesome-ML-SYS-Tutorial（RLHF System 开发笔记）
* [RLHF System 开发笔记](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/README-cn.md)
* [浅析以 OpenRLHF 为代表的 post-training 系统的计算流程](https://zhuanlan.zhihu.com/p/16370000391)
* [RL 系统深思：深入理解权重更新机制](https://zhuanlan.zhihu.com/p/1925210722704531547)

## oops 异常


## 神经网络
* http://udlbook.github.io/udlbook  深度学习中的算法背后的原理
* https://github.com/changyeyu/LLM-RL-Visualized  图解大模型算法
* https://www.rethink.fun  大模型核心技术和应用

## 分布式
* Ray-利用Ray进行大模型的数据处理、训练、推理和部署 [Ray rllib github地址](https://github.com/ray-project/ray/tree/master/rllib)
* [图解OpenRLHF中基于Ray的分布式训练流程](https://zhuanlan.zhihu.com/p/12871616401)

## vLLM
* [图解Vllm V1系列1：整体流程](https://zhuanlan.zhihu.com/p/1900126076279160869)

## Transformer 架构
* [Self-Attention（自注意力机制）](https://zhuanlan.zhihu.com/p/455399791)

## 流水线并行，数据并行和张量并行 (猛猿)

![mengyuan-megatron](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/tpdppp.png?v=1)
    
    1.节点内部 (8 卡)：使用 TP (张量并行)。因为 8 卡之间有 NVLink 高速互联，可以承受 TP 的高频通信，解决单卡存不下大层的问题。
    2.节点之间：使用 PP (流水线并行)。将模型层切分到不同的机器组上，减少跨机器的通信频率。
    3.整体集群：使用 DP (数据并行：模型复制，数据分片)。将上述的 "TP+PP" 组合视为一个大的“虚拟卡”，然后复制多份这样的组合，处理不同的数据批次，通过 DP 来扩大总吞吐量。
* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models]()
* [流水线并行（Pipeline Parallelism）](https://zhuanlan.zhihu.com/p/613196255)
* [数据并行上篇(DP, DDP与ZeRO)](https://zhuanlan.zhihu.com/p/617133971)
* [数据并行下篇( DeepSpeed ZeRO，零冗余优化)](https://zhuanlan.zhihu.com/p/618865052)
* [张量模型并行(TP)，Megatron-LM](https://zhuanlan.zhihu.com/p/622212228)
## megatron (猛猿)
![mengyuan-megatron](https://github.com/ztaoing/DeepRL_Steps/blob/main/image/mengyuan_megatron.png?v=4)
```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
* [Megatron官方手册](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/overview.html)
* [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed)
* [1、Megatron源码解读1，分布式环境初始化](https://zhuanlan.zhihu.com/p/629121480)
* [2、Megatron源码解读2，模型并行](https://zhuanlan.zhihu.com/p/634377071)
* [3、Megatron源码解读3，分布式混合精度训练](https://zhuanlan.zhihu.com/p/662700424)
* [4、DeepSpeed-Megatron MoE并行训练（原理篇）](https://zhuanlan.zhihu.com/p/681154742)
* [5、DeepSpeed-Megatron MoE并行训练（源码解读篇）](https://zhuanlan.zhihu.com/p/681692152)
* [【论文】Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
* [【论文】Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473 )
* [【论文】Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)


## 《动手学大模型》 
* [上海交大](https://github.com/Lordog/dive-into-llms)
* [昇腾开源文档中心](https://ascend.github.io/docs/index.html)
* 
### 大模型训练
* [1、训练数据准备](https://www.hiascend.com/developer/courses/detail/1871373977722732546)
* [2、大模型预训练](https://www.hiascend.com/developer/courses/detail/1872199134812995586)
* [2.1、预训练](https://www.hiascend.com/developer/courses/detail/1915241801172033538)
* [3、大模型推理](https://www.hiascend.com/developer/courses/detail/1872200627859406849)
* [4、大模型性能评估](https://www.hiascend.com/developer/courses/detail/1872202813553147905)
### 大模型推理
* [大模型微调-高级](https://www.hiascend.com/developer/courses/detail/1915246001570357250)
### 大模型评估
* [1-常用大模型评估指标](https://www.hiascend.com/developer/courses/detail/1872202813553147905)
* [2-基于C-Eval的大模型性能评估](https://www.hiascend.com/developer/courses/detail/1915237566367830017)

##  训练与推理

* [推理框架的切换成本,降低RL训推共卡开销：SGLang/vLLM的无缝切换实现与分析](https://zhuanlan.zhihu.com/p/2002748926185469778)
* [VMM：虚拟地址与物理地址之间的映射](https://github.com/CalvinXKY/BasicCUDA/tree/master/memory_opt/vmm)
* [训推角色切换与权重更新](https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/switch_role_update_weights.ipynb)
* [睡眠模式](https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md)
* [一文读懂vLLM显存管理：技术细节+优化思路](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247483759&idx=1&sn=419dcd4a4b0504a2dd6d1b1abf4f830a&scene=21&poc_token=HGF2lmmjrqicKBuswG6j7MhDdAJr1D9tO3loIbWq)

## vllm
* [vllm.llm_engine.model_executor]()
    - 是 vLLM 架构中连接“大脑”（调度与逻辑）与“肌肉”（实际计算）的关键组件
## Ray
* [ppo-ray(猛猿)]()
* [openrlhf-_initiate_actors ](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/openrlhf-_initiate_actors%20Ray%20%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%90%AF%E5%8A%A8%E5%99%A8%EF%BC%88Launcher%EF%BC%89%20%E7%9A%84%E6%A0%B8%E5%BF%83%E9%80%BB%E8%BE%91.ipynb)
    - 这段代码展示了如何利用 Ray 的 Placement Group (资源组) 来实现严格的资源隔离和分布式训练环境的搭建。
    - 创建 Master Actor (Rank 0)
    - 创建 Worker Actors (Rank 1 to N)
    - placement_group放置组
        - 资源束 (Bundle)
    - options() 动态配置任务所需的资源
* [openrlhf-LLMRayActor]()
    - 旨在利用 Ray 框架对 vLLM进行分布式推理封装
    - <font color='red'>init_process_group()它是连接 Ray 分布式框架与 PyTorch 分布式通信 (NCCL) 的关键桥梁。</font>
        - 在 OpenRLHF 这种混合训练框架中，我们需要将负责训练的 Deepspeed Actor 和负责推理的 vLLM Engine 纳入同一个通信组，以便在训练结束后将权重同步给推理引擎。
        - <font color='red'>训练</font>：Trainer (Actor)：使用 PyTorch/Deepspeed 进行梯度更新。
        - <font color='red'>推理</font>：Rollout (vLLM)：使用 vLLM 进行高效的文本生成。
* [openrlhf-ds_rank0与vllm_ranks之间的通讯（训练与推理之间的通信）]()
    - 它的作用是构建一个跨越“训练”与“推理”两个异构系统的统一分布式通信组（Process Group）
    - _broadcast_to_vllm() 方法的作用就是在这两个进程组之间搭建一座桥梁，将 DeepSpeed 侧更新后的权重“广播”给 vLLM 侧的所有进程
    - update_weight() 方法就是 vLLM 进程中的“接收器”，它监听来自 DeepSpeed 主进程的广播，并将接收到的权重加载到自己的模型中
* [openrlhf.utils.distributed_util]()
* [torch.distributed.broadcast]()
* [PPO-Actor/Critic Training]()
    - Step1：发送prompts，并从vllm_engine上收集(prompt, response)
    - Step2：从Ref/Reward/Critic上收集并处理exps
    - Step3: 确保将处理后的exps传送给Critic，并行执行Actor和Critic的训练
    - Step4：vllm_engine权重更新
* [openrlhf.utils.distributed_util]()
    - init_process_group(）
    - torch.distributed.init_process_group()[]
        - torch.distributed.init_process_group
        - torch.distributed.get_rank / get_world_size
        - torch.distributed.destroy_process_group
        - torch.distributed.barrier

* [ray-driver](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray-driver%E8%BF%9B%E7%A8%8B.ipynb)
* [ray核心api](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/Ray%E6%A0%B8%E5%BF%83API%E5%85%A8%E5%AE%B6%E6%A1%B6.ipynb)
    - [ray.put()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.put().ipynb)
    - [ray.wait()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.wait().ipynb)
    - [ray.get_actor()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.get_actor().ipynb)
    - [ray.shutdown()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.shutdown().ipynb)
    - [ray.kill](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.kill(actor).ipynb)
    - [ray.nodes()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.nodes()%20ray.cluster_resources()%20%20ray.timeline(filename).ipynb)
    - [ray.cluster_resources()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.nodes()%20ray.cluster_resources()%20%20ray.timeline(filename).ipynb)
    - [ray.available_resources()](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.nodes()%20ray.cluster_resources()%20%20ray.timeline(filename).ipynb)
    - [ray.timeline(filename)](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/ray/ray.nodes()%20ray.cluster_resources()%20%20ray.timeline(filename).ipynb)
## veRL

* [TaskRunner](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/TaskRunner.ipynb)
* [verl.single_controller.ray](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/verl.single_controller.ray.ipynb)
* [verl.trainer](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/verl.trainer.ipynb)
* [verl.utils](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/verl.utils.ipynb)
* [verl.workers](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/%20verl.workers.ipynb)
* [DataProto统一数据协议](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/DataProto%E7%BB%9F%E4%B8%80%E6%95%B0%E6%8D%AE%E5%8D%8F%E8%AE%AE(Data%20Protocol).ipynb)
* [self.actor_rollout_wg](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/self.actor_rollout_wg.ipynb)
* [verl-装饰器](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/verl%E8%A3%85%E9%A5%B0%E5%99%A8.ipynb)
* [verl-上下文管理器](https://github.com/ztaoing/DeepRL_Steps/blob/main/train_code/VERL/verl%E4%B8%8A%E4%B8%8B%E6%96%87%E7%AE%A1%E7%90%86%E5%99%A8.ipynb)
### verl设计

<div align="center">
  <img src="https://github.com/ztaoing/DeepRL_Steps/blob/main/image/ray.png?v=1"  />
</div>

[Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/README-cn.md)
* [深入浅出理解 verl 源码（Part 1）](https://zhuanlan.zhihu.com/p/1920751852749849692)
* [深入浅出理解 verl 源码——Rollout](https://zhuanlan.zhihu.com/p/1923349757566388159)
* [verl 参数速览](https://zhuanlan.zhihu.com/p/1925041836998783250)

## pytorch
* [pytorch官方手册](https://docs.pytorch.org/docs/stable/index.html)
* [pytorch examples](https://github.com/pytorch/examples/tree/main)
* [PyTorch显存管理介绍与源码解析（一）](https://zhuanlan.zhihu.com/p/680769942)
