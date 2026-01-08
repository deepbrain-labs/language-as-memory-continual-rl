# AGENTS.md

## Project Goals
- **Phase 0:** Reproducible infra and minimal end-to-end loop (State -> LLM -> Subgoal -> PPO Agent).

## Technical Constraints & Guidelines
- **Execution Environment:** Code must handle both `GPU_MODE` (Colab T4) and `CPU_MOCK_MODE` (Local dev).
  - Use `torch.cuda.is_available()` to guard heavy model loading.
  - Default to `MockPlanner` if no GPU is found.
- **Environment:** `minigrid` (specifically `MiniGrid-DoorKey-6x6-v0` and `MiniGrid-Empty-6x6-v0`).
- **RL Framework:** Start with `stable-baselines3` (SB3). Use `MultiInputPolicy` to handle `Dict` observations containing visual state and subgoal.
- **Subgoals:**
  - Use canonical discrete IDs (0-30).
  - Use structured tuples (Action, Color, Object) for reasoning.
  - Deterministic parser to map LLM text output to canonical tokens.
- **Seeds:** Ensure global reproducibility (Torch, Numpy, Random, Env).

## Directory Structure
- `src/`: Core logic.
- `notebooks/`: Colab notebooks (must import from `src`).
- `experiments/`: Logs and outputs.

## Workflow
- Notebooks should enable `!pip install -r requirements.txt` and then run.
- `00_poc_pipeline.ipynb` is the primary deliverable for Phase 0.
