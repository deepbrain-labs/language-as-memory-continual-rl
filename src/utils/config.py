import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

@dataclass
class RunConfig:
    """
    Configuration object for ensuring reproducibility and tracking experiment details.
    """
    seed: int
    env_id: str
    task_order: Union[List[str], str]
    model_revision: str = "main"
    prompt_version: str = "v1"
    
    def save(self, save_dir: str, filename: str = "run_config.json"):
        """Saves the configuration to a JSON file."""
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)
        print(f"RunConfig saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "RunConfig":
        """Loads a configuration from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
