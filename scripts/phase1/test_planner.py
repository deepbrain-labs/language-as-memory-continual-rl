import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from llm.planner import get_planner

def test_planner():
    print("=== LLM Planner Test ===")
    
    # Use Mock configs for speed/CPU, or auto-detect
    config = {
        'env': {'id': 'MiniGrid-DoorKey-6x6-v0'},
        'llm': {
            'model_name': 'microsoft/phi-2',
            'load_in_4bit': True,
            'use_lora': False,
            'temperature': 0.1,
            'max_new_tokens': 50,
            'mock_mode': 'auto' # Will use Mock if no GPU
        }
    }
    
    planner = get_planner(config)
    
    # Test Case 1: Start
    state1 = "You are carrying nothing. In the room, you see: yellow key, yellow door, green goal."
    subgoal1 = planner.generate_subgoal(state1)
    print(f"\nState: {state1}\nSubgoal: {subgoal1}")
    
    # Test Case 2: Has Key
    state2 = "You are carrying a yellow key. In the room, you see: yellow door, green goal."
    subgoal2 = planner.generate_subgoal(state2)
    print(f"\nState: {state2}\nSubgoal: {subgoal2}")
    
    # Test Case 3: Door Open
    state3 = "You are carrying a yellow key. In the room, you see: open yellow door, green goal."
    subgoal3 = planner.generate_subgoal(state3)
    print(f"\nState: {state3}\nSubgoal: {subgoal3}")
    
    print("\n✅ PHASE 1: Planner Test Passed.")

if __name__ == "__main__":
    test_planner()
