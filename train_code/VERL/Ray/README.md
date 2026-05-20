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
