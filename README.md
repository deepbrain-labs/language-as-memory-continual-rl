
# Language as Memory: Continual Reinforcement Learning

**DeepBrain Labs Research Project**

---

## Overview

This repository presents a research framework for exploring the intersection of language, memory, and continual reinforcement learning (RL). Our goal is to develop agents that leverage language as a structured memory system, enabling robust continual learning and reasoning in complex, partially observable environments.

The project is designed for reproducible, modular experimentation, supporting both rapid prototyping and rigorous evaluation. It is built and maintained by DeepBrain Labs, with a focus on advancing the state of the art in language-augmented RL.

---

## Key Features

- **Language-Augmented RL:** Integrates large language models (LLMs) as planners and memory modules within RL agents.
- **Continual Learning:** Supports lifelong learning scenarios, including subgoal discovery, curriculum learning, and robustness to distributional shifts.
- **Symbolic Subgoal Reasoning:** Utilizes structured, canonical subgoal representations for interpretable agent planning and analysis.
- **Reproducible Infrastructure:** Ensures experiment reproducibility across hardware (GPU/CPU), seeds, and environments.
- **Modular Design:** Clean separation of core logic, environment wrappers, LLM interfaces, and analysis tools.

---

## Project Goals

- **Phase 0:** Establish a minimal, reproducible end-to-end pipeline: State → LLM → Subgoal → PPO Agent.
- **Phase 1+:** Expand to more complex continual learning settings, richer subgoal spaces, and advanced LLM integration.

---

## Technical Highlights

- **Environments:** Built on `minigrid` (e.g., `MiniGrid-DoorKey-6x6-v0`, `MiniGrid-Empty-6x6-v0`).
- **RL Framework:** Utilizes `stable-baselines3` with `MultiInputPolicy` for handling multimodal observations.
- **Subgoal Representation:** Canonical discrete IDs (0-30) and structured tuples (Action, Color, Object) for deterministic parsing and reasoning.
- **LLM Integration:** Modular planner interface, with GPU/CPU fallback and mock support for local development.
- **Experiment Management:** All outputs and logs are stored in the `experiments/` directory for easy tracking and analysis.

---

## Directory Structure

```
├── src/                # Core logic and modules
│   ├── analysis/       # Analysis scripts and visualization tools
│   ├── configs/        # Default and experiment-specific configs
│   ├── data_gen/       # Data generation and trace utilities
│   ├── envs/           # Environment wrappers and noise models
│   ├── experiments/    # Training and evaluation scripts
│   ├── llm/            # LLM planner and RLHF modules
│   ├── ppo/            # PPO agent implementations
│   ├── rm/             # Reward model training
│   └── utils/          # Shared utilities (seeding, logging, etc.)
├── notebooks/          # Colab and Jupyter notebooks for experiments
├── experiments/        # Experiment logs and outputs
├── data/               # Datasets and expert traces
├── requirements.txt    # Python dependencies
└── README.md           # Project overview (this file)
```

---

## Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
# Clone the repository
$ git clone https://github.com/deepbrain-labs/language-as-memory-continual-rl.git
$ cd language-as-memory-continual-rl

# Install dependencies
$ pip install -r requirements.txt
```

### 2. Running Experiments

- **Colab/Jupyter:** Use the provided notebooks in `notebooks/` (e.g., `00_poc_pipeline.ipynb`) for interactive experimentation. Notebooks are designed to be self-contained and import from `src/`.
- **Scripts:** For full pipeline runs, use scripts in `src/experiments/` or the provided PowerShell script in `scripts/`.

### 3. Reproducibility

- All experiments are seed-controlled (Torch, Numpy, Random, Env).
- Hardware-aware: Automatically selects GPU or CPU mock planner based on availability.

---

## Citation

If you use this codebase or ideas in your research, please cite:

```
@misc{deepbrain2026languageasMemory,
	title={Language as Memory: Continual Reinforcement Learning},
	author={DeepBrain Labs},
	year={2026},
	url={https://github.com/deepbrain-labs/language-as-memory-continual-rl}
}
```

---

## Contact & Contributions

- For questions, open an issue or contact the DeepBrain Labs team.
- Contributions are welcome! Please submit a pull request or contact us to discuss research collaborations.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Advancing continual learning with language and memory.*