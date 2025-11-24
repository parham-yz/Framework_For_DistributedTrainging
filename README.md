# Block‑Wise Deep‑Learning Sandbox

This repository hosts a compact research playground for experimenting with block‑wise optimisation, distributed training, and instrumentation of neural networks. Every model exposes its layers through a `ModuleList` (`model.blocks`), making it straightforward to copy, swap, or analyse individual blocks while keeping the rest of the framework minimal and hackable.


## Highlights
- **Block awareness everywhere.** Architectures (feed‑forward, CNN, ResNet‑style, nanoGPT) ship with explicit block lists so training engines, measurement units, and plotting utilities can operate on sub‑structures.
- **Composable training frames.** `src/Buliding_Units/Model_frames.py` wires the centre model, distributed replicas, datasets, reporters, stoppers, and measurement units, letting you toggle between blockwise sequential and full‑model (`entire`) modes with a flag.
- **Plugin instrumentation.** Measurement units (memory footprint, finite‑difference Hessian, block interaction heat‑maps, …) are auto‑loaded from `measurements/Units.txt`; adding a new unit is as simple as registering its class name.
- **Batteries included datasets.** `src/Data/data_sets.py` covers MNIST variants, CIFAR‑10/100, SVHN, ImageNet/Mini‑ImageNet, IMDb, California Housing, and toy regressions so experiments stay focused on optimisation ideas rather than data plumbing.
- **Automation hooks.** Shell launchers under `lunch_files/` sweep step sizes and block counts, while `plot.py` aggregates JSON logs into publication‑ready figures.


## Agent Protocol (`Agents/AGENTS.md`)
The repository is operated by an `ml_research_agent` persona whose ground rules live in `Agents/AGENTS.md`. Before running anything, skim and update that file—it doubles as a lab notebook. Key expectations:

- **Persist learnings.** Append takeaways, pitfalls, and heuristics back into `Agents/AGENTS.md` after every meaningful run so the next iteration starts from the latest context.
- **Mirror metrics to disk.** Any metric printed to stdout per iteration/epoch must also be serialized under `ploting_data/<run_tag>/` (e.g., `.npy`, `.csv`, plus a compact `metrics_log.csv`) to keep plotting reproducible across scripts.
- **Deterministic plotting.** When using Matplotlib, cycle colors in the order `black → red → blue → brown → purple → green` before falling back to defaults; `plot.py` already follows this pattern.
- **Persona & workflow.** Operate like a senior research engineer: plan runs explicitly (dataset slices, layer configs, penalties, seeds), tee stdout to `_logs/<timestamp>.log`, keep data provenance clear, and summarise each run in Markdown (drop short notes in `figures/README.md` or similar).
- **Testing discipline.** Add focused tests under `tests/` for new solvers or optimisers and run `python -m pytest tests -q` before relying on a change. Smoking the MNIST baseline for 5 epochs is the default regression test when you touch training loops.


## Quickstart

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
# optional but recommended
python -m venv .venv && source .venv/bin/activate
```
`src/utils.py` dynamically installs `GPUtil` and `onnx` if they are missing, but pre‑installing through the requirements file keeps runs deterministic.

### 2. Launch a training run
`src/dol1.py` exposes all hyper‑parameters as CLI flags through `utils.parse_arguments`. Typical invocation:

```bash
python -m src.dol1 \
  --model ResNet34-bi \
  --dataset_name cifar10 \
  --training_mode blockwise_sequential \
  --step_size 5e-3 \
  --batch_size 256 \
  --rounds 10000 \
  --K 10 \
  --cuda_core 0 \
  --config "[128,64,32,32,32]" \
  --report_sampling_rate 20 \
  --measurement_sampling_rate 400 \
  --reports_dir rcnn_cifar10
```

Key knobs:
- `training_mode`: `blockwise_sequential`, `entire`, or `ploting` (exports ONNX snapshots).
- `dataset_name`: see the supported values in `src/Data/data_sets.py` (MNIST variants, CIFARs, SVHN, ImageNet/Mini-ImageNet, IMDb, California Housing, toy ones).
- `config`: JSON string interpreted by the chosen model loader (e.g., block widths for CNN/ResNet variants).
- `K`, `communication_delay`, `beta`, `n_workers`: control local steps per round, simulated latency, smoothing, and distributed worker counts for blockwise modes.
- `gamma`, `prox_lambda`, `block_accuracy_factor`: tune Algorithm 1 by blending updates, setting the proximal regularizer (scalar or per-block list), and defining the $\delta_i = \text{factor}\cdot \lambda_i \|\hat{x}^{(i)}_{t+1} - x^{(i)}_t\|$ stopping threshold for the inner solvers.

### 3. Inspect run artefacts
- **Reports** land in `reports/<reports_dir>/R*.txt` (auto-created). Each log contains the hashed hyper‑parameters, timestamped progress, and termination status via `utils.Reporter`.
- **Measurements** are appended to `measurements/<name>_log.txt` by each active plugin, and expensive routines such as the Hessian block interaction estimator populate `figures/Measurements/Hessian_Measurement/`.
- **Metrics serialization** must mirror console output under `ploting_data/<run_tag>/` (per the agent protocol). The plotting scripts treat that directory as ground truth when re-generating curves.
- **Plots**: run `python plot.py` (or call `plot_results`/`plot_with_variance` inside a notebook) to sweep `saved_logs/` and materialise comparison figures under `figures/`.


## Experiment Orchestration

### Training frames & engines
- `src/Buliding_Units/Model_frames.py` constructs task-specific frames (`ImageClassifier_frame_blockwise`, `ImageClassifier_frame_entire`, `Regression_frame_blockwise`, `Distributed_frame`) that own the centre model, the distributed copies (one per block), optimisers, stoppers, and reporters.
- `src/Optimizer_Engines/BCD_engine.py` houses the concrete training loops:
  1. `train_blockwise_sequential` – sequential proximal block updates plus the Algorithm 1 damping step.
  2. `train_entire` – vanilla whole-model training for baselines.
  Both loops share `log_progress`, which evaluates on a “big” batch at the cadence given by `report_sampling_rate`.

### Architectures
- `src/Architectures/feedforward_nn.py` – fully-connected networks (including CNN variants) plus ensembles that concatenate sub-model blocks.
- `src/Architectures/resnets.py` – ResNet‑18/34 recreation that stays weight-compatible with torchvision while preserving block metadata.
- `src/Architectures/nanoGPT.py` – compact GPT implementation usable as another block‑wise architecture.
- `src/Architectures/Models.py` – loader helpers (`load_resnet18`, `load_feedforward_ensemble`, etc.) that keep scripts model-agnostic.

### Data utilities
`src/Data/data_sets.py` unifies dataset downloads and custom subsets:
- Image classification: `mnist`, `mini_mnist`, `mini_mnist_8chanel`, `mnist_flat`, `cifar10`, `cifar100`, `svhn`, `imagenet`, `mini_imagenet`.
- Regression & tabular: `ones`, `california_housing` (requires `scikit-learn`).
- Text: `imdb`.
Each dataset shares preprocessing transforms so hyper-parameter sweeps remain comparable.

### Measurement & logging plugins
- `src/Buliding_Units/MeasurementUnit.py` defines the plugin interface plus built-ins such as `Working_memory_usage`, `Hessian_measurement`, and `HessianBlockInteractionMeasurement`. New units should subclass `MeasurementUnit`, implement `measure(frame)`, and register their class name in `measurements/Units.txt`.
- `src/Buliding_Units/StopperUnit.py` provides patience-based (`EarlyStoppingStopper`) and threshold-based (`TargetStopper`, `AccuracyTargetStopper`) early stopping hooks.
- `measurement_sampling_rate` throttles how often each unit runs; heavy routines like Hessians usually run every few hundred rounds.

### Batch launchers
`lunch_files/lunchV2.sh` and `lunch_files/mnist_cnn.sh` sweep combinations of step sizes and block counts (`K`) across GPUs, handle cleanup on `SIGINT`, and store outputs under the `reports_dir` configured inside each script. Use them as templates for larger grids.


## Repository Layout

```
├── Agents/AGENTS.md          # ml_research_agent playbook / lab notebook
├── lunch_files/              # bash launchers for large experiment grids
├── measurements/             # plugin logs + Units.txt registry
├── plot.py                   # plotting + log aggregation helpers
├── requirements.txt
├── src/
│   ├── Architectures/        # feedforward, CNN, ResNet, nanoGPT, loaders
│   ├── Buliding_Units/       # Model frames, measurement + stopper units
│   ├── Data/                 # dataset loaders & synthetic generators
│   ├── Optimizer_Engines/    # blockwise / entire training loops
│   ├── dol1.py               # CLI entry point (invoked via python -m src.dol1)
│   └── utils.py              # reporters, CLI parsing, ONNX export, helpers
├── tests/                    # regression tests for block copying, Hessians, etc.
└── llms_tools/               # scratch space for LLM-generated experiment notes
```

Directories such as `reports/`, `figures/`, `ploting_data/`, `_logs/`, and `saved_logs/` are created at runtime and should remain in `.gitignore` unless a figure is curated for documentation.


## Testing & Quality Gates
- Run `python -m pytest tests -q` before trusting changes. The suite checks block isolation, copy semantics, Hessian utilities, and distributed update logic.
- For new optimisation ideas, add a regression test that hits the smallest reproducible case (e.g., verify conjugate-gradient convergence within `N` steps on a synthetic system).
- When touching training loops, smoke test a short MNIST/CIFAR run (5 epochs) and stash the log under `_logs/`.


## Extending the Framework
- **New measurement**: subclass `MeasurementUnit`, emit metrics + figures, register the class in `measurements/Units.txt`, and remember to write measurements to `ploting_data/<run_tag>/`.
- **New optimiser/engine**: drop a function in `src/Optimizer_Engines/`, accept a `Frame` instance, and hook it into `src/dol1.py`.
- **New architecture**: expose `model.blocks` as a `ModuleList` and add a loader in `src/Architectures/Models.py`.
- **Document everything**: when you uncover a pitfall, log it to `Agents/AGENTS.md`, mention it in the run summary, and (when helpful) codify it as a test.

Have fun hacking—this codebase is intentionally small so you can trace every tensor in an afternoon while still experimenting with serious distributed-training ideas.
