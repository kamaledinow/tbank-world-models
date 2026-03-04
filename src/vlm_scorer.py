from __future__ import annotations

<<<<<<< codex/create-demo-project-for-world-model-and-vlm-20y1fn
from typing import Any, Sequence
=======
from typing import Sequence
>>>>>>> main
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

<<<<<<< codex/create-demo-project-for-world-model-and-vlm-20y1fn
    def _to_embedding_tensor(self, output: Any) -> torch.Tensor:
        """Convert various transformers outputs into a [B, D] embedding tensor.

        Some environments/forks may return model outputs (e.g. BaseModelOutputWithPooling)
        where vanilla transformers would return features tensor.
        """
        if isinstance(output, torch.Tensor):
            return output

        # Common CLIP named outputs
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            # fallback: mean-pool sequence tokens
            return output.last_hidden_state.mean(dim=1)

        raise TypeError(f"Unsupported CLIP output type: {type(output)}")

    def _normalize(self, emb: torch.Tensor) -> torch.Tensor:
        denom = emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb / denom

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        toks = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        raw = self.model.get_text_features(**toks)
        emb = self._to_embedding_tensor(raw)
        return self._normalize(emb)
=======
    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        toks = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        emb = self.model.get_text_features(**toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb
>>>>>>> main

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
<<<<<<< codex/create-demo-project-for-world-model-and-vlm-20y1fn
            raw = self.model.get_image_features(**inputs)
            img_emb = self._normalize(self._to_embedding_tensor(raw))
=======
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
>>>>>>> main
            sims = (img_emb @ self.text_emb.T).squeeze(-1)
            sims_all.append(sims.detach().cpu())

        sims_flat = torch.cat(sims_all, dim=0).numpy()

        out = np.zeros((num_traj,), dtype=np.float32)
        idx = 0
        for k, ln in enumerate(lengths):
            out[k] = float(np.max(sims_flat[idx : idx + ln])) if ln > 0 else 0.0
            idx += ln
        return out
