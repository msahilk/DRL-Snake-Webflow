import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

import gymnasium as gym
from gymnasium import spaces

# =============================
# SnakeTestEnv — DRL test Snake
# =============================
# Grid-based snake with built-in "fault injection" switches to emulate bugs/regressions.
# Focus: Automated testing personas (collector vs explorer) and rich metrics.
#
# Board:
#   0 = empty, 1 = snake body, 2 = food, 3 = wall (outer border)
#
# Actions (relative to current heading):
#   0 = turn left, 1 = go straight, 2 = turn right
#
# Observation (11-D vector, normalized):
#   [danger_ahead, danger_left, danger_right,
#    dir_up, dir_down, dir_left, dir_right,
#    food_dx_norm, food_dy_norm,
#    norm_steps, norm_unique]  in [0..1] or [-1..1] where noted.
#
# Reward personas:
# - collector:   +10 apple, -10 death, -0.01 per step (time pressure)
# - explorer:   +0.2 per *new tile*, -0.005 on revisits, +2 apple, +30 when ~100% covered, no time pressure
#
# Fault toggles (for experiments):
# - fault_invisible_wall: a random inner cell blocks passage but renders as empty (in metrics, count wall_hits)
#
# Metrics (returned via info):
#   steps, apples, length, died, cause, wall_hits, self_hits, turns, unique_cells, coverage_ratio,
#   time_since_last_food, starvation_softlock, bug_events
#
# Termination:
#   - death (wall or self) or step_limit reached
#
# Stable-Baselines3 compatible.

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L (row, col)
LEFT = 0
STRAIGHT = 1
RIGHT = 2

class SnakeTestEnv(gym.Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(
        self,
        size: int = 12,
        step_limit: int = 1000,
        reward_mode: str = "collector",
        seed: Optional[int] = None,
        fault_invisible_wall: bool = False,
    ):
        super().__init__()
        assert reward_mode in ("collector", "explorer")
        self.size = size
        self.step_limit = step_limit
        self.reward_mode = reward_mode
        self.fault_invisible_wall = fault_invisible_wall
        self._ate_last_step = False
        self._unique_cells_prev = 0

        # Spaces
        self.action_space = spaces.Discrete(3)
        low = np.array([0, 0, 0,  0, 0, 0, 0,  -1, -1,  0, 0], dtype=np.float32)
        high= np.array([1, 1, 1,  1, 1, 1, 1,   1,  1,  1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # RNG
        self.np_random = np.random.default_rng(seed)
        self._seed = seed

        # State
        self.grid = None
        self.snake: List[Tuple[int, int]] = []
        self.dir_idx: int = 1  # start moving right
        self.food: Tuple[int, int] = (0, 0)
        self.steps = 0
        self.turns = 0
        self.apples = 0
        self.wall_hits = 0
        self.self_hits = 0
        self.unique_cells = set()
        self.time_since_last_food = 0
        self.pending_food_delay = 0
        self.starvation_softlock = 0
        self.bug_events = 0
        self.invisible_wall_cell: Optional[Tuple[int, int]] = None
        self.last_cause = "alive"

    def seed(self, seed: Optional[int] = None):
        self.np_random = np.random.default_rng(seed)
        self._seed = seed

    # ------------ helpers ------------
    def _empty_grid(self):
        g = np.zeros((self.size, self.size), dtype=np.int8)
        g[0, :] = 3; g[-1, :] = 3; g[:, 0] = 3; g[:, -1] = 3  # border walls
        return g

    def _random_empty_cell(self):
        # exclude walls and snake cells and invisible wall cell
        tries = 0
        while True:
            r = self.np_random.integers(1, self.size - 1)
            c = self.np_random.integers(1, self.size - 1)
            if (r, c) in self.snake:
                tries += 1
                if tries > 500: return (1, 1)
                continue
            if self.invisible_wall_cell and (r, c) == self.invisible_wall_cell:
                tries += 1
                if tries > 500: return (1, 1)
                continue
            return (int(r), int(c))

    def _place_food(self):
        if self.pending_food_delay > 0:
            self.pending_food_delay -= 1
            return
        self.food = self._random_empty_cell()
        self.grid[self.food] = 2

    def _set_invisible_wall(self):
        if self.fault_invisible_wall:
            self.invisible_wall_cell = self._random_empty_cell()
        else:
            self.invisible_wall_cell = None

    def _init_snake(self):
        center = (self.size // 2, self.size // 2)
        self.snake = [center, (center[0], center[1] - 1), (center[0], center[1] - 2)]
        self.dir_idx = 1  # right

    def _danger_flags(self) -> Tuple[float, float, float]:
        # Check immediate collisions for straight/left/right if we move next step.
        flags = []
        for rel in (LEFT, STRAIGHT, RIGHT):
            dir_idx = (self.dir_idx + (-1 if rel == LEFT else (0 if rel == STRAIGHT else 1))) % 4
            dr, dc = DIRECTIONS[dir_idx]
            head_r, head_c = self.snake[0]
            nr, nc = head_r + dr, head_c + dc
            danger = 0.0
            # walls (including invisible wall)
            if nr <= 0 or nr >= self.size - 1 or nc <= 0 or nc >= self.size - 1:
                danger = 1.0
            elif self.invisible_wall_cell and (nr, nc) == self.invisible_wall_cell:
                danger = 1.0
            elif (nr, nc) in self.snake[:-1]:
                danger = 1.0
            flags.append(danger)
        return tuple(flags)

    def _obs(self) -> np.ndarray:
        d_ahead, d_left, d_right = self._danger_flags()
        dir_onehot = [0, 0, 0, 0]
        dir_onehot[self.dir_idx] = 1
        head_r, head_c = self.snake[0]
        dx = (self.food[1] - head_c) / (self.size - 1)
        dy = (self.food[0] - head_r) / (self.size - 1)
        norm_steps = min(1.0, self.steps / max(1, self.step_limit))
        norm_unique = min(1.0, len(self.unique_cells) / float((self.size - 2) * (self.size - 2)))
        return np.array([
            d_ahead, d_left, d_right,
            *dir_onehot,
            float(dx), float(dy),
            norm_steps, norm_unique
        ], dtype=np.float32)

    def _rotate(self, action: int):
        if action == LEFT:
            self.dir_idx = (self.dir_idx - 1) % 4
            self.turns += 1
        elif action == RIGHT:
            self.dir_idx = (self.dir_idx + 1) % 4
            self.turns += 1
        # STRAIGHT leaves direction as is

    def _move(self) -> Tuple[bool, str]:
        dr, dc = DIRECTIONS[self.dir_idx]
        head_r, head_c = self.snake[0]
        nr, nc = head_r + dr, head_c + dc

        # wall collision (including invisible wall cell)
        if nr <= 0 or nr >= self.size - 1 or nc <= 0 or nc >= self.size - 1 or (self.invisible_wall_cell and (nr, nc) == self.invisible_wall_cell):
            self.wall_hits += 1
            return True, "wall"

        # self collision (with optional fault skip)
        if (nr, nc) in self.snake[:-1]:
            self.self_hits += 1
            return True, "self"

        # move snake
        ate = (nr, nc) == self.food
        self._ate_last_step = ate

        self.snake.insert(0, (nr, nc))
        if ate:
            self.apples += 1
            self.time_since_last_food = 0
            # clear food
            self.grid[self.food] = 0
            self._place_food()
        else:
            self.time_since_last_food += 1
            tail = self.snake.pop()  # move forward
            # clear tail cell in grid
            if 0 < tail[0] < self.size - 1 and 0 < tail[1] < self.size - 1:
                self.grid[tail] = 0

        # draw head
        self.grid[nr, nc] = 1
        self.unique_cells.add((nr, nc))
        return False, "alive"

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self.grid = self._empty_grid()
        self._init_snake()
        for r, c in self.snake:
            self.grid[r, c] = 1
        self._set_invisible_wall()
        self.apples = 0
        self.steps = 0
        self.turns = 0
        self.wall_hits = 0
        self.self_hits = 0
        self.unique_cells = set(self.snake)
        self.time_since_last_food = 0
        self.pending_food_delay = 0
        self.starvation_softlock = 0
        self.last_cause = "alive"
        self._ate_last_step = False
        self._unique_cells_prev = len(self.unique_cells)

        # place initial food
        self._place_food()
        return self._obs(), self._info()

    def step(self, action: int):
        self.steps += 1
        self._rotate(int(action))

        done, cause = self._move()
        self.last_cause = cause
        reward = 0.0

        # --- Reward shaping per persona ---
        if self.reward_mode == "collector":
            reward -= 0.01  # light time pressure
            if self._ate_last_step:
                reward += 10.0

        elif self.reward_mode == "explorer":
            # reward ONLY for first-time tile visits (novelty)
            new_tile = 1 if len(self.unique_cells) > self._unique_cells_prev else 0
            if new_tile:
                reward += 0.3       # tune 0.1–0.3 to taste
            else:
                reward -= 0.005     # tiny revisit penalty
            if self._ate_last_step:
                reward += 1       # smaller than collector; focus stays on coverage

        if done:
            reward -= 10.0

        # starvation softlock heuristic
        if self.time_since_last_food > max(40, self.size * 3):
            self.starvation_softlock = 1

        terminated = done
        truncated = self.steps >= self.step_limit

        obs = self._obs()
        info = self._info()

        # Explorer: finish & big bonus when ~all inner tiles visited
        if self.reward_mode == "explorer" and info["coverage_ratio"] >= 0.99 and not terminated and not truncated:
            reward += 30.0
            terminated = True
            self.last_cause = "covered"
            info = self._info()  # refresh if you want, but metrics already reflect coverage

        # update novelty baseline for next step
        self._unique_cells_prev = info["unique_cells"]

        return obs, reward, terminated, truncated, info

    def _info(self) -> Dict[str, Any]:
        walkable = (self.size - 2) * (self.size - 2)
        return {
            "steps": self.steps,
            "apples": self.apples,
            "length": len(self.snake),
            "died": int(self.last_cause in ("wall", "self")),
            "cause": self.last_cause,
            "wall_hits": self.wall_hits,
            "self_hits": self.self_hits,
            "turns": self.turns,
            "unique_cells": len(self.unique_cells),
            "coverage_ratio": len(self.unique_cells) / walkable,
            "time_since_last_food": self.time_since_last_food,
            "starvation_softlock": self.starvation_softlock,
            "bug_events": self.bug_events,
        }

    def render(self) -> str:
        # ANSI rendering
        chars = {0: " ", 1: "o", 2: "*", 3: "#"}  # snake body 'o', food '*'
        s = []
        for r in range(self.size):
            line = []
            for c in range(self.size):
                ch = chars[int(self.grid[r, c])]
                if (r, c) == self.snake[0]:
                    ch = "O"  # head
                # do not visually reveal invisible wall fault
                line.append(ch)
            s.append("".join(line))
        return "\n".join(s)
