import argparse
import yaml
import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from src.envs import make_env
from src.wrappers import FacingGoalHintWrapper, LearnedHintWrapper
from src.abstraction import AbstractionNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_experiment(config):
    seed = config.get('seed', 0)
    set_seed(seed)
    env_id = config.get('env_id', 'MiniGrid-Empty-5x5-v0')
    env = make_env(env_id)

    # choose mode: baseline, handhint, learnedhint
    mode = config.get('mode', 'baseline')
    if mode == 'handhint':
        env = FacingGoalHintWrapper(env)
    elif mode == 'learnedhint':
        # load a trained abstraction model from path
        model_path = config.get('abstraction_path')
        if model_path is None:
            raise ValueError("abstraction_path required for learnedhint mode")
        # load model (assumes same input dim)
        obs, _ = env.reset()
        input_dim = obs.shape[0]
        abstraction = AbstractionNet(input_dim)
        abstraction.load_state_dict(torch.load(model_path, map_location='cpu'))
        env = LearnedHintWrapper(env, abstraction)

    # IMPORTANT: initialize PPO AFTER wrapping
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
)

    total_timesteps = config.get('timesteps', 10000)
    model.learn(total_timesteps=total_timesteps)
    save_dir = config.get('save_dir', './models')
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"ppo_{mode}_seed{seed}"))
    print("Saved PPO model to", save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    run_experiment(config)
