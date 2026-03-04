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
    k_cand, horizon = action_seqs.shape
    state = RSSMState(
        deter=start_state.deter.repeat(k_cand, 1),
        stoch=start_state.stoch.repeat(k_cand, 1),
    )
    total = torch.zeros(k_cand, device=device)
    imagined_frames = [[] for _ in range(k_cand)]

    for t in range(horizon):
        a_oh = one_hot(action_seqs[:, t], model.action_dim).to(device)
        state = model.imagine_step(state, a_oh)
        obs_pred, rew_pred, _ = model.decode(state)
        total += (config.gamma**t) * rew_pred

        if use_vlm:
            frames = (obs_pred.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            frames = np.transpose(frames, (0, 2, 3, 1))
            for i in range(k_cand):
                imagined_frames[i].append(frames[i])

    if use_vlm and vlm_scorer is not None:
        # Batch scoring all trajectories at once is dramatically faster than per-candidate calls.
        vlm_scores_np = vlm_scorer.score_trajectories(imagined_frames, image_batch_size=64)
        vlm_scores = torch.from_numpy(vlm_scores_np).to(device=device, dtype=total.dtype)
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
