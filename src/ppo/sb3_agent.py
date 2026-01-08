import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import torch
import torch.nn as nn

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for MiniGrid + Subgoal.
    It receives the Dict observation.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Image processing (CNN)
        # MiniGrid images are (H, W, 3) but usually small (7x7 for partial view, or grid size).
        # We assume standard numeric image input.
        n_input_channels = observation_space["image"].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing a forward pass
        with torch.no_grad():
            sample_image = torch.as_tensor(observation_space["image"].sample()[None]).float().permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_image).shape[1]

        # Subgoal processing (Embedding)
        # input is (1,), long
        self.subgoal_embedding = nn.Embedding(31, 16) # 30 tokens + padding/extra

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process Image
        # SB3 automatically moves tensors to device, but we need to ensure correct shape/type
        image = observations["image"].permute(0, 3, 1, 2).float() # (B, C, H, W)
        img_feats = self.cnn(image)

        # Process Subgoal
        subgoal = observations["subgoal"].long().squeeze(1) # (B,)
        sub_feats = self.subgoal_embedding(subgoal)

        # Concatenate
        combined = torch.cat([img_feats, sub_feats], dim=1)

        return self.linear(combined)

def create_agent(env, config):
    """
    Creates and returns a PPO agent configured for the environment.
    """

    # We use MultiInputPolicy because our observation is a Dict
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    agent = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=config['rl']['learning_rate'],
        n_steps=config['rl']['n_steps'],
        batch_size=config['rl']['batch_size'],
        n_epochs=config['rl']['n_epochs'],
        gamma=config['rl']['gamma'],
        gae_lambda=config['rl']['gae_lambda'],
        clip_range=config['rl']['clip_range'],
        ent_coef=config['rl']['ent_coef'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=config['experiment']['device']
    )

    return agent
