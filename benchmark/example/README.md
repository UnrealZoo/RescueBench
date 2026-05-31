# Rescue Benchmark Quick Start

This folder is the fastest path for a new user. After the conda environment is ready, you can verify the benchmark, run a baseline, or plug in your own `solution.py` without editing the core benchmark code.

## 1. Enter The Environment

```bash
cd /path/to/RescueBench
conda activate <your-env>
```

## 2. Verify The Benchmark

Run a random policy first. This checks environment startup, task loading, wrappers, scoring, and result writing without model weights.

```bash
python benchmark/experiment.py --model random --levels 0 --episodes 1 --no-collision
```

Results are written to:

```text
benchmark_results/random/
```

## 3. Run The Editable Solution Template

```bash
python benchmark/example/experiment.py
```

This launcher loads:

```text
benchmark/example/solution.py
```

To connect a model, edit `AlgSolution`:

```python
class AlgSolution:
    def reset(self, reference_text=None, reference_image=None, info=None):
        ...

    def predicts(self, observation, picked, info=None):
        return {
            "angular": 0.0,
            "velocity": 50.0,
            "viewport": 0,
            "interaction": 0,
        }
```

Default inputs:

- `observation`: base64 PNG image.
- `reference_image`: base64 PNG reference image, or `None`.
- `reference_text`: task text.
- `picked`: whether the injured person is currently being carried.
- `info`: state dict with keys such as `Pose`, `task_phase`, `target_pose`, `level`, and `point_id`.

Use raw numpy observations instead of base64 with:

```bash
--solution-input-format raw
```

## 4. Run Your Solution From The Unified Entrypoint

```bash
python benchmark/experiment.py \\
  --model solution \\
  --solution benchmark/example/solution.py \\
  --levels 0 \\
  --episodes 1
```

Interaction control modes:

```bash
# The model returns interaction=3 for carry and interaction=4 for drop.
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --passthrough

# The model only navigates; the benchmark state machine inserts carry/drop/open_door.
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --no-passthrough
```

## 5. Available Models

The unified entrypoint currently supports these `--model` names:

```text
random       # Random policy for smoke tests
solution     # User-provided solution.py adapter
apex（yolo）  # Apex baseline, requires its workspace/weights
r2zeroshot   # R2ZeroShot baseline, requires its workspace/weights
citywalker   # CityWalker baseline
seepointfly  # SeePointFly multi-agent baseline
vint         # ViNT navigation baseline
nomad        # NOMAD navigation baseline
omninav      # OmniNav baseline
uninavid     # Uni-NaVid baseline
uni_navid    # Compatibility alias for uninavid
```

If a baseline environment, weights, and config paths are already set up, run it by name:

```bash
python benchmark/experiment.py --model citywalker --levels 0 --episodes 1
```

Model registration and defaults live in:

```text
benchmark/agents/factory.py    # AGENT_REGISTRY: model name -> Agent class
benchmark/agents/profiles.py   # MODEL_PROFILES: resolution, passthrough, and defaults
```

You can also list CLI options with:

```bash
python benchmark/experiment.py --help
```

## 6. Common Options

```bash
# Save render frames
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --render

# Save every frame and generate videos after the run
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --render --save-frame-every 1 --save-video

# Record trajectories
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --enable-trajectory

# Resume from an existing jsonl result file
python benchmark/experiment.py --model solution --solution benchmark/example/solution.py --levels 0 --episodes 1 --resume-jsonl benchmark_results/solution/benchmark_solution_xxx.jsonl
```

Render frames are written to:

```text
benchmark_results/<model>/_render_frames/latest_frame.jpg
```

Watch the latest frame in another terminal:

```bash
python benchmark/frame_viewer.py --path benchmark_results/solution/_render_frames/latest_frame.jpg
```

## 7. Main Files

```text
benchmark/experiment.py              # Recommended unified entrypoint
benchmark/example/experiment.py      # Minimal editable experiment config
benchmark/example/solution.py        # User model I/O template
benchmark/agents/solution_agent.py   # Adapter that loads solution.py
benchmark/run_*.py                   # Baseline-specific compatibility launchers
```
