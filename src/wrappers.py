import gymnasium as gym
import numpy as np
import torch

class FacingGoalHintWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # Update observation space (+1 hint)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([old_space.low, np.array([0])]),
            high=np.concatenate([old_space.high, np.array([1])]),
            dtype=old_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_hint(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_hint(obs), reward, terminated, truncated, info

    def _add_hint(self, obs):
        agent_dir = self.env.unwrapped.agent_dir
        agent_pos = self.env.unwrapped.agent_pos

        goal_pos = None
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                obj = self.env.unwrapped.grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    goal_pos = (x, y)
                    break

        hint = 0
        if goal_pos is not None:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]

            if (agent_dir == 0 and dx > 0) or \
               (agent_dir == 1 and dy > 0) or \
               (agent_dir == 2 and dx < 0) or \
               (agent_dir == 3 and dy < 0):
                hint = 1

        return np.concatenate([obs, np.array([hint], dtype=obs.dtype)])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_hint(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_hint(obs), reward, terminated, truncated, info

    def _add_hint(self, obs):
        agent_dir = self.env.unwrapped.agent_dir
        agent_pos = self.env.unwrapped.agent_pos

        goal_pos = None
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                obj = self.env.unwrapped.grid.get(x, y)
                if obj is not None and obj.type == "goal":
                    goal_pos = (x, y)
                    break

        hint = 0
        if goal_pos is not None:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]

            if (agent_dir == 0 and dx > 0) or \
               (agent_dir == 1 and dy > 0) or \
               (agent_dir == 2 and dx < 0) or \
               (agent_dir == 3 and dy < 0):
                hint = 1

        return np.concatenate([obs, np.array([hint], dtype=obs.dtype)])

class LearnedHintWrapper(gym.Wrapper):
    def __init__(self, env, abstraction_model):
        super().__init__(env)
        self.abstraction_model = abstraction_model
        self.abstraction_model.eval()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def _augment(self, obs):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            hint = self.abstraction_model(x).item()
            hint = 1 if hint > 0.5 else 0
        return np.concatenate([obs, np.array([hint], dtype=obs.dtype)])
