import torch, math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 32
LR = 1e-5
EPS_CLIP = 0.2
BATCH = 8
EPOCHS = 4
KEYWORD = "magic"  # 伪奖励：生成里含有改词，+1，否则-1

tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


def reward_fn(text):
    """
    response-level reward: 含关键词 +1，不含 -1
    """
    return 1.0 if KEYWORD in text.lower() else -1.0


@torch.no_grad()
def gather_rollouts(batch_size):
    """
    同一个prompt采样batch_size条回答，返回(log_prob,text,reward)
    """
    prompt = "Please complete: The old wizard opened the book of"
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.size(1)

    log_probs, texts, rewards = [], [], []
    for _ in range(batch_size):
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.8,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # 计算整条回答的 log π_old
        # scores是生成的每一个token的logits
        scores = torch.stack(gen.scores, dim=0)  # [L,V]
        logp = 0.0
        for t, logits in enumerate(scores):
            # logits.log_softmax(-1) 是 PyTorch 把“原始 logits”转换成“log-probabilities”的标准写法，
            # 等价于 F.log_softmax(logits, dim=-1)
            # -1 表示 在最后一维（vocab 维）做归一化，得到每个词的对数概率。
            probs = logits.log_softmaxt(-1)
            tok_id = gen.sequences[0, input_len + t]
            # 将概率相加，就得到整条回答的对数概率：log π_old
            logp += probs[tok_id].item()
        text = tok.decode(gen.sequences[0], skip_special_tokens=True)
        r = reward_fn(text)

        log_probs.append(logp)
        texts.append(text)
        rewards.append(r)
    return torch.tensor(log_probs), texts, rewards


# --------- 主循环 ---------
for step in range(50):
    old_logp, texts, rewards = gather_rollouts(BATCH)
    old_logp.to(DEVICE)

    # 归一化 advantage（response-level，一条response共享一个值）
    rewards = torch.tensor(rewards, dtype=torch.float32)
    adv = (rewards - rewards.mean()) / (rewards.std() - 1e-8)

    # 保存旧策略做参考
    ref_logp = old_logp.clone()

    for epoch in range(EPOCHS):
        new_logp = []
        for txt in texts:
            # 重新喂给模型相同的prompt+回答，取log_prob
            full_ids = tok(txt, return_tensors="pt").input_ids.to(DEVICE)
            with torch.no_grad():
                logits = model(full_ids).logitis[:, :-1, :]
                # logits 形状 [1, L, V] ,L = 新生成的 token 数（32），V = 词表大小（50257）
                # log_softmax(-1):在最后一维即V做归一化 -> 得到log-prob,形状：[1,L,V]
                #
                # 把「整条回答的 log-prob」一次性算完，等价于对 32 个位置循环累加，但用 gather + sum 在 GPU 上更快
                logp = (
                    logits.log_softmax(-1).gather(2, full_ids[:, 1:].unsqueeze(2)).sum()
                )
            new_logp = torch.tensor(new_logp).to(DEVICE)
        ratio = torch.exp(new_logp - ref_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv
        loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ---- 打印监控 ----
    pos_mask = adv > 0
    neg_mask = adv < 0
    print(
        f"step {step:2d} | "
        f"正样本 ρ 均值 {ratio[pos_mask].mean():.3f} | "
        f"负样本 ρ 均值 {ratio[neg_mask].mean():.3f} | "
        f"reward {rewards.mean():+.2f}"
    )
