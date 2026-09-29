import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List
from tqdm import trange
from encoding import decode_seq

# (ActorCritic 类保持不变，此处省略，请保留原文件中的定义)
class ActorCritic(nn.Module):
    def __init__(self, seq_len, action_dim, d_model=64, n_head=4, n_layers=2):
        super(ActorCritic, self).__init__()
        self.embedding = nn.Embedding(4, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.actor_head = nn.Linear(d_model, action_dim)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        emb = self.embedding(x) + self.pos_embedding
        feat = self.transformer(emb)
        feat_mean = feat.mean(dim=1)
        logits = self.actor_head(feat_mean)
        value = self.critic_head(feat_mean)
        return logits, value

# -----------------------------
# PPO Agent
# -----------------------------
class PPOAgent:
    def __init__(
        self,
        seq_len: int,
        action_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
        d_model: int = 64,
    ):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        
        self.net = ActorCritic(seq_len, action_dim, d_model=d_model).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(self, obs: torch.LongTensor):
        self.net.eval()
        obs = obs.unsqueeze(0).to(self.device)
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def _compute_gae(self, rewards, values, dones, last_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - float(dones[t])
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        returns = adv + np.array(values, dtype=np.float32)
        return adv, returns

    def update(self, batch):
        self.net.train()
        obs = torch.as_tensor(batch["obs"], dtype=torch.long, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = obs.size(0)
        inds = np.arange(N)
        losses = []

        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, N, self.minibatch_size):
                mb_idx = inds[start:start + self.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_oldlog = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                logits, values = self.net(mb_obs)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = values.squeeze(-1)
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    # -----------------------------
    # 训练循环 (支持 Pre-existing Header)
    # -----------------------------
    def rollout_and_learn(
        self,
        env,
        steps_per_update: int = 1024,
        updates: int = 200,
        print_every: int = 10,
        log_path: str = "train_log.csv",
    ):
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, done_buf, val_buf = [], [], []
        
        obs = env.reset()
        episode_return = 0.0
        ep_returns: List[float] = []

        ep_counter = 1 # 从 1 开始，因为 0 留给了 WT
        pbar = trange(1, updates + 1, desc="Training PPO", leave=True)
        current_loss = 0.0
        


        # 如果文件已存在（说明 run_train_te 已经写了 WT 行），我们需要知道列的顺序！
        # 但 CSV 模块读这个比较麻烦。
        # 简单策略：我们依然用 sorted(metrics.keys())，因为 run_train_te 也是这么用的。
        # 只要两边都用 sorted，顺序就是一致的。

        for upd in pbar:
            obs_buf.clear(); act_buf.clear(); logp_buf.clear()
            rew_buf.clear(); done_buf.clear(); val_buf.clear()

            for t in range(steps_per_update):
                action, logprob, value = self.select_action(obs)
                
                next_obs, reward, done, info = env.step(action)
                metrics = info.get("metrics", {}) # 拿到指标字典

                obs_buf.append(obs.detach().cpu().numpy())
                act_buf.append(action)
                logp_buf.append(logprob)
                rew_buf.append(float(reward))
                done_buf.append(float(done))
                val_buf.append(value)

                obs = next_obs
                episode_return += reward

                if done:
                    ep_returns.append(episode_return)
                    seq_str = decode_seq(obs)
                    score = float(reward)

                    # Write row using fixed LOG_COLUMNS from run_train_te
                    from run_train_te import LOG_COLUMNS, _build_row
                    row = _build_row(upd, ep_counter, score, metrics, seq_str)
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    ep_counter += 1
                    obs = env.reset()
                    episode_return = 0.0

            # ... (PPO Update 逻辑不变) ...
            with torch.no_grad():
                last_obs = obs.unsqueeze(0).to(self.device)
                _, last_value = self.net(last_obs)
                last_value = float(last_value.item())

            advantages, returns = self._compute_gae(rew_buf, val_buf, done_buf, last_value)
            obs_np = np.stack(obs_buf, axis=0)
            
            batch = {
                "obs": obs_np,
                "actions": np.array(act_buf),
                "logprobs": np.array(logp_buf, dtype=np.float32),
                "returns": returns.astype(np.float32),
                "advantages": advantages.astype(np.float32),
                "values": np.array(val_buf, dtype=np.float32),
            }
            loss = self.update(batch)
            current_loss = loss

            avg_return = np.mean(ep_returns[-10:]) if ep_returns else 0.0
            pbar.set_description(f"Upd {upd} | Loss {loss:.4f} | AvgReward {avg_return:.3f}")