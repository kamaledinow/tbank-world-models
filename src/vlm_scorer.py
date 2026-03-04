from __future__ import annotations

from typing import Sequence
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPScorer:
    def __init__(self, text_goal: str, device: str = "cpu", model_name: str = "openai/clip-vit-base-patch32"):
        self.device = torch.device(device)
        self.text_goal = text_goal
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.text_emb = self._encode_text(text_goal)

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        toks = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        emb = self.model.get_text_features(**toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def score_frames(self, frames: Sequence[np.ndarray]) -> float:
        """Score a single trajectory (list of frames)."""
        batch_scores = self.score_trajectories([frames])
        return float(batch_scores[0])

    @torch.no_grad()
    def score_trajectories(self, trajectories: Sequence[Sequence[np.ndarray]], image_batch_size: int = 64) -> np.ndarray:
        """Score many imagined trajectories efficiently.

        Returns one scalar per trajectory: max_t(sim(frame_t, text_goal)).
        """
        num_traj = len(trajectories)
        if num_traj == 0:
            return np.zeros((0,), dtype=np.float32)

        lengths = [len(tr) for tr in trajectories]
        total_frames = sum(lengths)
        if total_frames == 0:
            return np.zeros((num_traj,), dtype=np.float32)

        flat_images = [Image.fromarray(frame) for tr in trajectories for frame in tr]
        sims_all = []

        for i in range(0, len(flat_images), image_batch_size):
            chunk = flat_images[i : i + image_batch_size]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ self.text_emb.T).squeeze(-1)
            sims_all.append(sims.detach().cpu())

        sims_flat = torch.cat(sims_all, dim=0).numpy()

        out = np.zeros((num_traj,), dtype=np.float32)
        idx = 0
        for k, ln in enumerate(lengths):
            out[k] = float(np.max(sims_flat[idx : idx + ln])) if ln > 0 else 0.0
            idx += ln
        return out
