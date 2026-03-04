from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict


class TinyGridGoalEnv:
    """Simple 2D grid environment with RGB observations.

    Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
    Goal: move the agent onto the green target.
    """

    def __init__(self, size: int = 6, max_steps: int = 40, tile_size: int = 8, seed: int = 0):
        self.size = size
        self.max_steps = max_steps
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed)
        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.goal_pos = np.array([size - 1, size - 1], dtype=np.int64)
        self.steps = 0

    @property
    def action_space_n(self) -> int:
        return 5

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self.agent_pos = self.rng.integers(0, self.size, size=2)
        self.goal_pos = self.rng.integers(0, self.size, size=2)
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = self.rng.integers(0, self.size, size=2)
        return self.render()

    def step(self, action: int) -> StepResult:
        self.steps += 1
        move = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),
        }[int(action)]
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        hit_goal = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if hit_goal else -0.01
        done = hit_goal or self.steps >= self.max_steps
        return StepResult(obs=self.render(), reward=reward, done=done, info={"hit_goal": hit_goal})

    def render(self) -> np.ndarray:
        h = self.size * self.tile_size
        w = self.size * self.tile_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = np.array([25, 25, 25], dtype=np.uint8)

        # Draw grid lines
        img[:: self.tile_size, :, :] = 50
        img[:, :: self.tile_size, :] = 50

        # Goal (green)
        gx, gy = self.goal_pos
        self._fill_tile(img, gx, gy, np.array([0, 200, 0], dtype=np.uint8))

        # Agent (blue)
        ax, ay = self.agent_pos
        self._fill_tile(img, ax, ay, np.array([40, 120, 255], dtype=np.uint8))

        # If on goal, blend color
        if np.array_equal(self.agent_pos, self.goal_pos):
            self._fill_tile(img, ax, ay, np.array([0, 255, 255], dtype=np.uint8))

        return img

    def _fill_tile(self, img: np.ndarray, x: int, y: int, color: np.ndarray) -> None:
        ts = self.tile_size
        x0, y0 = x * ts + 1, y * ts + 1
        x1, y1 = (x + 1) * ts - 1, (y + 1) * ts - 1
        img[x0:x1, y0:y1] = color
