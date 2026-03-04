from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.env import TinyGridGoalEnv
from src.planner import PlanConfig, one_hot, random_shooting_action
from src.vlm_scorer import CLIPScorer
from src.world_model import RSSM


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs).float().to(device) / 255.0
    return x.permute(2, 0, 1)


def collect_random_episodes(env: TinyGridGoalEnv, episodes: int, device: torch.device):
    data = []
    for ep in range(episodes):
        obs = env.reset(seed=ep)
        done = False
        ep_data = []
        while not done:
            action = np.random.randint(env.action_space_n)
            step = env.step(action)
            ep_data.append((obs, action, step.reward, step.done, step.obs))
            obs = step.obs
            done = step.done
        data.append(ep_data)
    return data


def train_world_model(model: RSSM, dataset, device: torch.device, epochs: int = 15, lr: float = 3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in trange(epochs, desc="train"):
        random.shuffle(dataset)
        total_loss = 0.0
        for episode in dataset:
            state = model.init_state(batch_size=1, device=device)
            loss = 0.0
            for obs, action, reward, done, next_obs in episode:
                obs_t = preprocess(next_obs, device).unsqueeze(0)
                emb = model.encoder(obs_t)
                a_oh = F.one_hot(torch.tensor([action], device=device), num_classes=model.action_dim).float()
                state, kl = model.observe_step(state, a_oh, emb)
                obs_pred, rew_pred, done_logit = model.decode(state)
                rec_loss = F.mse_loss(obs_pred, obs_t)
                rew_t = torch.tensor([reward], dtype=torch.float32, device=device)
                done_t = torch.tensor([float(done)], dtype=torch.float32, device=device)
                rew_loss = F.mse_loss(rew_pred, rew_t)
                done_loss = F.binary_cross_entropy_with_logits(done_logit, done_t)
                step_loss = rec_loss + 0.2 * kl.mean() + rew_loss + 0.2 * done_loss
                loss = loss + step_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            total_loss += float(loss.item())
    return model


@torch.no_grad()
def infer_state_from_obs(model: RSSM, obs: np.ndarray, device: torch.device):
    x = preprocess(obs, device).unsqueeze(0)
    emb = model.encoder(x)
    state = model.init_state(batch_size=1, device=device)
    a = F.one_hot(torch.tensor([4], device=device), num_classes=model.action_dim).float()
    state, _ = model.observe_step(state, a, emb)
    return state


@torch.no_grad()
def run_policy(env, model, device, policy_name, scorer=None, plan_cfg=None, seed=0, save_gif=None):
    obs = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    frames = [obs]
    while not done:
        if policy_name == "random":
            action = np.random.randint(env.action_space_n)
        else:
            state = infer_state_from_obs(model, obs, device)
            action = random_shooting_action(
                model=model,
                start_state=state,
                config=plan_cfg,
                use_vlm=(policy_name == "wm_vlm"),
                vlm_scorer=scorer,
            )
        step = env.step(action)
        obs = step.obs
        total_reward += step.reward
        done = step.done
        frames.append(obs)
    if save_gif:
        imageio.mimsave(save_gif, frames, fps=5)
    success = int(total_reward > 0.0)
    return total_reward, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train-episodes", type=int, default=120)
    p.add_argument("--train-epochs", type=int, default=12)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--goal-text", type=str, default="agent at the green goal")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    env = TinyGridGoalEnv(size=6, max_steps=35, tile_size=8, seed=args.seed)
    model = RSSM(action_dim=env.action_space_n).to(device)

    dataset = collect_random_episodes(env, episodes=args.train_episodes, device=device)
    train_world_model(model, dataset, device=device, epochs=args.train_epochs)

    scorer = CLIPScorer(text_goal=args.goal_text, device=args.device)
    plan_cfg = PlanConfig(horizon=10, num_candidates=24, vlm_weight=2.0)

    metrics = {}
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    for policy in ["random", "wm_no_vlm", "wm_vlm"]:
        returns, succs = [], []
        for ep in range(args.eval_episodes):
            gif_path = out_dir / f"{policy}_ep{ep}.gif" if ep == 0 else None
            ret, succ = run_policy(
                env=env,
                model=model,
                device=device,
                policy_name=policy,
                scorer=scorer,
                plan_cfg=plan_cfg,
                seed=1000 + ep,
                save_gif=str(gif_path) if gif_path else None,
            )
            returns.append(ret)
            succs.append(succ)
        metrics[policy] = {
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(succs)),
            "episodes": args.eval_episodes,
        }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
