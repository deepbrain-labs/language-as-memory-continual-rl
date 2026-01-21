import gymnasium as gym
import sys
import os
import numpy as np
from minigrid.core.actions import Actions

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from envs.wrappers import SubgoalWrapper
from utils.subgoal_parser import parse_subgoal
from llm.planner import get_planner

class SimpleBot:
    """
    A heuristic bot that can execute subgoals by navigating the grid.
    Used for Phase 1 POC to validate the pipeline before training PPO.
    """
    def __init__(self, env):
        self.env = env # Unwrapped or Wrapper that gives access to grid

    def get_action(self, subgoal_tuple):
        """
        Returns a primitive action to progress towards the subgoal.
        """
        action_type, color, obj_type = subgoal_tuple
        
        # Access underlying grid
        # wrapper -> env -> unwrapped
        grid = self.env.unwrapped.grid
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        
        if action_type == "no_op":
            return Actions.done
            
        # Find target
        target_pos = None
        
        # Special case: Door might be open or closed
        # If 'open' action, we look for closed door.
        # If 'goto' door, we look for any door.
        
        best_dist = 999
        
        for i in range(grid.width):
            for j in range(grid.height):
                cell = grid.get(i, j)
                if not cell: continue
                
                match = False
                if cell.type == obj_type:
                    if color == "any" or cell.color == color:
                        match = True
                
                if match:
                    # distance
                    dist = abs(agent_pos[0] - i) + abs(agent_pos[1] - j)
                    if dist < best_dist:
                        best_dist = dist
                        target_pos = (i, j)

        if not target_pos:
            return Actions.forward # Random explore if target not found
            
        # Navigation Logic (Simple BFS or Manhattan heuristic towards target)
        # For Phase 1 POC in empty room/doorkey, simple heuristic works:
        # Align X, then Align Y.
        
        ax, ay = agent_pos
        tx, ty = target_pos
        
        # If we are adjacent and need to interact
        is_adjacent = (abs(ax - tx) + abs(ay - ty)) == 1
        
        # Check facing
        # 0: right, 1: down, 2: left, 3: up
        dx = tx - ax
        dy = ty - ay
        
        desired_dir = None
        if dx == 1: desired_dir = 0
        elif dx == -1: desired_dir = 2
        elif dy == 1: desired_dir = 1
        elif dy == -1: desired_dir = 3
        
        is_facing = (agent_dir == desired_dir)
        
        if is_adjacent:
            if not is_facing:
                return Actions.left # Turn until facing
            else:
                if action_type == "pick":
                    return Actions.pickup
                elif action_type == "open":
                    return Actions.toggle
                elif action_type == "goto":
                    return Actions.done # We arrived? Or just stand there.
                    # Usually "goto" means be adjacent.
        
        # Movement
        if not is_facing:
            # Simple turn logic: assume we want to go in desired_dir
            # For POC we can just try random turns or specific if we want to be smart.
            # Let's simple spin.
            return Actions.left
        else:
            # Check if front is clear
            front_cell = grid.get(ax+dx, ay+dy)
            if front_cell is None or front_cell.type in ['open', 'goal', 'lava']: # passable
                 return Actions.forward
            elif front_cell.type == 'door' and front_cell.is_open:
                 return Actions.forward
            else:
                 return Actions.left # Blocked, turn
                 
        return Actions.forward

def run_chain_poc():
    print("=== Phase 1: End-to-End Chain POC ===")
    
    # 1. Setup
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = SubgoalWrapper(env)
    
    # Planner
    config = {
        'env': {'id': 'MiniGrid-DoorKey-6x6-v0'},
        'llm': {
            'model_name': 'microsoft/phi-2',
            'load_in_4bit': True,
            'use_lora': False,
            'temperature': 0.1,
            'max_new_tokens': 50,
            'mock_mode': 'auto' 
        }
    }
    planner = get_planner(config)
    bot = SimpleBot(env)
    
    obs, _ = env.reset(seed=42)
    max_steps = 50
    
    print("\nStarting Episode...")
    
    current_subgoal = "None"
    
    for step in range(max_steps):
        # 1. Get Description
        desc = env.get_text_description()
        
        # 2. Plan (Simulate "Encoder -> LLM")
        # Optimization: Only replan if subgoal completed or new episode
        # For POC, let's replan every few steps or if 'subgoal_completed' flag is set?
        # Actually, let's just plan every step for observability in logs.
        
        new_subgoal_text = planner.generate_subgoal(desc)
        (action, color, obj), sg_id = parse_subgoal(new_subgoal_text)
        
        if new_subgoal_text != current_subgoal:
            print(f"\n[Step {step}] New Plan: {new_subgoal_text}")
            current_subgoal = new_subgoal_text
            env.set_subgoal((action, color, obj), sg_id)
            
        # 3. Execute (Bot)
        # The bot needs to know the CURRENT target.
        primitive_action = bot.get_action((action, color, obj))
        
        # 4. Step
        obs, reward, terminated, truncated, info = env.step(primitive_action)
        
        # Log
        # print(f"  Action: {primitive_action} | Reward: {reward}")
        
        if terminated or truncated:
            print(f"\nEpisode finished at step {step}. Success: {reward > 0}")
            break
            
    print("\n✔ PHASE 1: POC Chain Execution Complete.")

if __name__ == "__main__":
    run_chain_poc()
