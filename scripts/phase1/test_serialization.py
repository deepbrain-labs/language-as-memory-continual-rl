import gymnasium as gym
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper

def test_serialization():
    print("=== State Serialization Test ===")
    
    # Setup Env
    env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="rgb_array")
    env = SubgoalWrapper(env)
    
    # 1. Determinism Check
    print("Checking determinism...")
    obs, _ = env.reset(seed=42)
    text1 = env.get_text_description()
    
    obs, _ = env.reset(seed=42)
    text2 = env.get_text_description()
    
    assert text1 == text2, "❌ Serialization is not deterministic!"
    print(f"✔ Determinism passed.\nText: {text1}")
    
    # 2. Content Check
    print("\nChecking content...")
    # DoorKey-6x6 should have a key and a door.
    assert "key" in text1, "❌ Key not found in description"
    assert "door" in text1, "❌ Door not found in description"
    assert "carrying nothing" in text1, "❌ Carrying status incorrect"
    
    print("✔ Content validation passed.")
    print("\n✅ PHASE 1: Serialization Test Passed.")

if __name__ == "__main__":
    test_serialization()
