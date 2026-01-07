import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

def make_env(env_id="MiniGrid-Empty-5x5-v0"):
    """
    Create and return a MiniGrid environment wrapped with FlatObsWrapper.
    """
    env = gym.make(env_id)
    env = FlatObsWrapper(env)
    return env
