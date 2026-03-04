from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


class Encoder(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, 128 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.fc(feat).view(-1, 128, 6, 6)
        return self.deconv(x)


class RSSM(nn.Module):
    def __init__(self, action_dim: int, deter_dim: int = 128, stoch_dim: int = 32, emb_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.encoder = Encoder(emb_dim=emb_dim)
        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior = nn.Linear(deter_dim, 2 * stoch_dim)
        self.posterior = nn.Linear(deter_dim + emb_dim, 2 * stoch_dim)

        feat_dim = deter_dim + stoch_dim
        self.decoder = Decoder(feat_dim)
        self.reward_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.done_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return RSSMState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def _dist_params(self, h: torch.Tensor, posterior_emb: torch.Tensor | None = None):
        prior_stats = self.prior(h)
        prior_mean, prior_logstd = prior_stats.chunk(2, dim=-1)
        prior_std = F.softplus(prior_logstd) + 1e-4
        if posterior_emb is None:
            return prior_mean, prior_std, prior_mean, prior_std
        post_stats = self.posterior(torch.cat([h, posterior_emb], dim=-1))
        post_mean, post_logstd = post_stats.chunk(2, dim=-1)
        post_std = F.softplus(post_logstd) + 1e-4
        return prior_mean, prior_std, post_mean, post_std

    def observe_step(self, prev: RSSMState, action: torch.Tensor, obs_emb: torch.Tensor):
        x = torch.cat([prev.stoch, action], dim=-1)
        h = self.gru(x, prev.deter)
        prior_mean, prior_std, post_mean, post_std = self._dist_params(h, obs_emb)
        eps = torch.randn_like(post_mean)
        stoch = post_mean + post_std * eps
        state = RSSMState(deter=h, stoch=stoch)
        kl = self._kl_normal(post_mean, post_std, prior_mean, prior_std)
        return state, kl

    def imagine_step(self, prev: RSSMState, action: torch.Tensor):
        x = torch.cat([prev.stoch, action], dim=-1)
        h = self.gru(x, prev.deter)
        prior_mean, prior_std, _, _ = self._dist_params(h, None)
        eps = torch.randn_like(prior_mean)
        stoch = prior_mean + prior_std * eps
        return RSSMState(deter=h, stoch=stoch)

    def features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def decode(self, state: RSSMState):
        feat = self.features(state)
        obs = self.decoder(feat)
        rew = self.reward_head(feat).squeeze(-1)
        done_logit = self.done_head(feat).squeeze(-1)
        return obs, rew, done_logit

    @staticmethod
    def _kl_normal(m1, s1, m2, s2):
        var1 = s1.pow(2)
        var2 = s2.pow(2)
        kl = torch.log(s2 / s1) + (var1 + (m1 - m2).pow(2)) / (2.0 * var2) - 0.5
        return kl.sum(dim=-1)
