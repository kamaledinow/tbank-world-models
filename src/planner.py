from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from src.world_model import RSSM, RSSMState


@dataclass
class PlanConfig:
    horizon: int = 12
    num_candidates: int = 64
    gamma: float = 0.99
    vlm_weight: float = 2.0


def one_hot(actions: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=n).float()


def evaluate_action_sequences(
    model: RSSM,
    start_state: RSSMState,
    action_seqs: torch.Tensor,
    use_vlm: bool,
    vlm_scorer=None,
    config: PlanConfig = PlanConfig(),
):
    device = start_state.deter.device
    K, H = action_seqs.shape
    # tile start state
    state = RSSMState(
        deter=start_state.deter.repeat(K, 1),
        stoch=start_state.stoch.repeat(K, 1),
    )
    total = torch.zeros(K, device=device)
    imagined_frames = [[] for _ in range(K)]

    for t in range(H):
        a_oh = one_hot(action_seqs[:, t], model.action_dim).to(device)
        state = model.imagine_step(state, a_oh)
        obs_pred, rew_pred, _ = model.decode(state)
        total += (config.gamma**t) * rew_pred
        if use_vlm:
            frames = (obs_pred.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
            for k in range(K):
                imagined_frames[k].append(frames[k])

    if use_vlm and vlm_scorer is not None:
        vlm_scores = torch.tensor([vlm_scorer.score_frames(frames) for frames in imagined_frames], device=device)
        total = total + config.vlm_weight * vlm_scores

    return total


def random_shooting_action(
    model: RSSM,
    start_state: RSSMState,
    config: PlanConfig,
    use_vlm: bool = False,
    vlm_scorer=None,
):
    action_seqs = torch.randint(
        low=0,
        high=model.action_dim,
        size=(config.num_candidates, config.horizon),
        device=start_state.deter.device,
    )
    scores = evaluate_action_sequences(
        model=model,
        start_state=start_state,
        action_seqs=action_seqs,
        use_vlm=use_vlm,
        vlm_scorer=vlm_scorer,
        config=config,
    )
    best = torch.argmax(scores)
    return int(action_seqs[best, 0].item())
