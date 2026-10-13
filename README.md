# Some Entertaining Projects 

A compact experimental AI playground for learning, prototyping, and testing ideas across three focused sub-projects:

- Maze solving and visual benchmarking
- Tiny agent reasoning and tool use
- Tiny world model / Dreamer-inspired learning

This repository is intended for research, study, and playful experimentation. It is designed to be easy to read, easy to run, and easy to extend for AI education, algorithm comparison, and world-model experiments.

## Why this repository exists

This project combines several AI learning themes in one place:

- Maze navigation and search algorithms
- Visual maze parsing from text, image, and PDF inputs
- Fog-of-war online planning
- Tool-using agents and memory-based planning
- Neural policy behavior cloning
- World-model learning and dream training
- Experimentation with model-based reasoning and decision making

The goal is not a production app, but a research sandbox for understanding how planning, perception, and learning interact.

## Repository structure

```text
MiniProject/
├── .gitignore
├── .python-version
├── LICENSE                  # Keep this file if your GitHub repo already created one
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   └── miniproject/
│       ├── __init__.py
│       ├── maze/
│       │   ├── app.py                  # Streamlit visual interface
│       │   ├── dataset.py              # Maze dataset utilities
│       │   ├── fog.py                  # Fog-of-war environment and online sensing
│       │   ├── generator.py            # Maze generation utilities
│       │   ├── grid.py                 # Maze grid parsing and representations
│       │   ├── main.py                 # CLI benchmark runner
│       │   ├── model.py                # Neural policy / Maze model logic
│       │   ├── online_solvers.py       # Online pathfinding under partial observability
│       │   ├── solvers.py              # BFS, DFS, Dijkstra, A*, Bidirectional A*, JPS, neural solver
│       │   ├── train_policy.py         # Neural policy training utilities
│       │   ├── vision.py               # Vision-based maze parsing
│       │   ├── visualizer.py           # Text/UI visualizers
│       │   ├── data/
│       │   │   ├── maze.txt            # Example maze file
│       │   │   ├── *.png               # Maze images
│       │   │   ├── *.jpg               # Maze images
│       │   │   └── *.pdf               # Maze PDFs (if used)
│       │   └── solved/
│       │       └── *.png               # Generated solved-output files
│       ├── tiny_agent/
│       │   ├── agent.py                # Agent loop: perceive -> think -> act
│       │   ├── environment.py          # Multi-room toy environment
│       │   ├── main.py                 # Demo of three planning paradigms
│       │   ├── memory.py               # Belief state and episodic memory
│       │   ├── neural_policy.py        # MLP-based policy + trainer
│       │   ├── planner.py              # Symbolic planner + ReAct-style LLM planner
│       │   ├── tools/
│       │   │   ├── actions.py          # Look / move / pickup / drop / open tools
│       │   │   ├── base.py             # Tool API definitions
│       │   └── __init__.py
│       └── tiny_world_model/
│           ├── dataset.py              # Rollouts and transitions
│           ├── dreamer.py              # Dreamer-style actor-critic trainer
│           ├── env.py                  # GridWorld environment
│           ├── main.py                 # World model + dream training demo
│           ├── model.py                # Tiny world model backbone
│           ├── planner.py              # Imagined-rollout planner
│           ├── trainer.py              # World model training loop
│           ├── tools/
│           │   ├── metrics.py          # Fidelity and quality metrics
│           │   └── visualizer.py       # Visual reporting utilities
│           └── __init__.py
```

## Project parts

### 1) Maze project

The maze package is the most complete part of the repository and acts as a benchmark suite for pathfinding and decision-making.

It contains:

- Grid parsing and maze normalization
- Classic search algorithms: BFS, DFS, Dijkstra, A*, Bidirectional A*, JPS
- Neural-policy-based solving
- Performance comparison and benchmark output
- Fog-of-war environments with online replanning
- Vision and OCR-like parsing for maze layouts from image/PDF inputs
- Visualization via terminal output and Streamlit app

Typical run patterns:

```bash
# Load a maze from a plain-text file
uv run python -m miniproject.maze.main --file src/miniproject/maze/data/maze.txt

# Solve a maze from an image or PDF
uv run python -m miniproject.maze.main --image src/miniproject/maze/data/maze.png

# Show a larger map in the terminal
uv run python -m miniproject.maze.main --file src/miniproject/maze/data/maze.txt --show-map

# Run with fog-of-war mode (online planner benchmark)
uv run python -m miniproject.maze.main --file src/miniproject/maze/data/maze.txt --fog --animate
```

The project can also be launched through the Streamlit UI:

```bash
uv run streamlit run src/miniproject/maze/app.py
```

Notes:

- Custom maze files should be placed inside `src/miniproject/maze/data/`.
- Generated solution visuals can be saved to `src/miniproject/maze/solved/`.
- Sample maze data is included as examples, but your own challenge maps can be added there.

### 2) Tiny agent project

The tiny agent package explores three different cognitive styles:

1. Symbolic planner
2. Neural policy planner
3. ReAct-style LLM planner

It uses a toy multi-room environment where the agent must carry an item to the target room while reasoning over observations, memory, and tools.

Main building blocks:

- `MultiRoomEnvironment`: a simple environment with rooms, doors, items, and inventory
- `TinyAgent`: the main control loop
- `AgentMemory`: stores history, beliefs, visited rooms, and known objects
- `BasePlanner`: planning interface
- `GoalDirectedPlanner`: rule-based symbolic planner
- `NeuralPolicyPlanner`: policy network trained by expert imitation
- `LLMPlanner`: ReAct-style planner with optional external API integration
- Tool layer: look, move, pickup, drop, and open door operations

Run it with:

```bash
uv run python -m miniproject.tiny_agent.main
```

This project is useful for understanding:

- how memory changes decisions
- how rule-based systems compare to learned policies
- how tool use and planning interact
- how an LLM can generate action plans from structured context

### 3) Tiny world model project

This module explores learning a compact dynamics model for a grid world and then using it for planning and decision-making.

It includes:

- `GridWorldEnv`: a minimal grid world with obstacles and goal rewards
- `TinyWorldModel`: approximates state transition dynamics via a learned model
- `WorldModelTrainer`: trains the transition model on collected experience
- `DreamActorCritic`: actor-critic policy for imagined rollout optimization
- `DreamerTrainer`: trains in a latent imagined space
- `ImaginedRolloutPlanner`: uses model rollouts to make decisions
- Reward and fidelity evaluation utilities

Run it with:

```bash
uv run python -m miniproject.tiny_world_model.main
```

This is an educational world-model project to explore:

- model-based reinforcement learning
- latent imagination
- actor-critic learning in simulated rollouts
- reduced-latency action selection from an internal predictor

## Data and output conventions

The repository is intentionally lightweight, but it benefits from a clear data layout.

### Recommended locations for data

```text
src/miniproject/maze/data/
    - place custom maze files here
    - accepted formats: .txt, .png, .jpg, .jpeg, .pdf

src/miniproject/maze/solved/
    - generated solved visualizations and results
    - keep this directory free of private or sensitive artifacts
```

Use the `data/` folder for inputs and the `solved/` folder for generated outputs. This keeps experiments organized and makes it easier to compare multiple runs.

### Data hygiene guidance

Because this project is research-oriented and may involve custom worlds, private scenarios, and generated outputs, keep the following out of source control:

- secret keys and API tokens
- local config files with environment data
- private test mazes
- generated solution images, debug files, and local benchmark outputs
- data created during experimentation that is not meant to be shared

The GitHub `.gitignore` file in this repository is designed to help with that.

## Installation and environment

This project uses Python and UV.

Install dependencies:

```bash
uv sync
```

If you want to use a local environment directly:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

The project metadata is configured in `pyproject.toml` and is set up for Python 3.13+.

## Typical workflows

### Run solver benchmarks

```bash
uv run python -m miniproject.maze.main --file src/miniproject/maze/data/maze.txt
```

### Run the agent demo

```bash
uv run python -m miniproject.tiny_agent.main
```

### Run the world-model demo

```bash
uv run python -m miniproject.tiny_world_model.main
```

### Start the visual maze app

```bash
uv run streamlit run src/miniproject/maze/app.py

```

## Contribution

This is primarily an educational and experimental project. Contributions are welcome if they improve:

- clarity of the demos
- algorithm correctness
- documentation
- benchmark quality
- training and visualization tools

## Summary

This repository is a compact AI research lab for:

- solving mazes with classical and modern methods
- comparing symbolic and learned agent strategies
- experimenting with model-based decision making
- exploring thinking, planning, memory, and environment modeling

It is intentionally simple, readable, and extensible so it can be used as a study project, a toy benchmark, or a seed for bigger AI experiments.
