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
        pil_images = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        img_emb = self.model.get_image_features(**inputs)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        sims = (img_emb @ self.text_emb.T).squeeze(-1)
        # encourage at least one good future state
        return float(torch.max(sims).cpu().item())
