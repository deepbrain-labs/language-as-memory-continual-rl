import sys
import os
import yaml
import torch
import numpy as np
import minigrid
import gymnasium as gym

# Ensure src is in path
sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.utils.subgoal_parser import parse_subgoal
from src.llm.planner import get_planner
from src.envs.wrappers import SubgoalWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds

def run_pipeline():
    print("=== Starting Phase 0 Verification Pipeline ===")

    # 1. Load Config
    with open("src/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print(f"Config loaded. Experiment ID: {config['experiment']['id']}")

    # 1.5 Set Global Seeds
    set_global_seeds(config['experiment']['seed'])

    # 2. Setup Logger
    logger = ExperimentLogger(save_dir=f"experiments/runs/{config['experiment']['id']}")
    logger.log_text(0, "Pipeline started.")

    # 3. Init Env
    env_id = config['env']['id']
    print(f"Initializing Environment: {env_id}")
    env = gym.make(env_id, render_mode="rgb_array")
    env = SubgoalWrapper(env)

    # 4. Init Planner
    print("Initializing Planner...")
    planner = get_planner(config)

    # 5. Reset Env & Get State
    obs, info = env.reset(seed=config['experiment']['seed'])
    state_text = env.get_text_description()
    print(f"Initial State Description: {state_text}")
    logger.log_text(0, f"State: {state_text}")

    # 6. Generate Subgoal (Mock or Real)
    print("Querying Planner for Subgoal...")
    subgoal_text = planner.generate_subgoal(state_text)
    print(f"Planner Output: {subgoal_text}")
    logger.log_text(0, f"Subgoal Text: {subgoal_text}")

    # 7. Parse Subgoal
    subgoal_tuple, subgoal_id = parse_subgoal(subgoal_text)
    print(f"Parsed Subgoal: {subgoal_tuple} -> ID: {subgoal_id}")

    # 8. Set Subgoal in Env
    env.set_subgoal(subgoal_tuple, subgoal_id)

    # 9. Init Agent
    print("Initializing PPO Agent...")
    agent = create_agent(env, config)

    # 10. Execute Agent (Short Loop)
    print("Executing Agent for a few steps...")

    # Since we are not training, we just predict actions
    # We need to reshape obs for SB3 if it expects vectorized input,
    # but PPO.predict handles single env instances usually if passed correctly.
    # However, SB3 models wrap environments in DummyVecEnv automatically during training,
    # but here we are using the raw env.

    total_reward = 0
    for step in range(10):
        action, _ = agent.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Log step
        print(f"Step {step+1}: Action={action}, Reward={reward}, Done={terminated or truncated}")

        if terminated or truncated:
            print("Episode finished early.")
            break

    logger.log_metrics(0, {"total_reward": total_reward})

    print("=== Verification Complete ===")

if __name__ == "__main__":
    run_pipeline()
