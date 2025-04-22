# Block‑Wise Deep‑Learning Sandbox

This repository is a **small research playground** for block‑wise optimisation and
distributed training of neural‑network models.  We wanted an environment that is

* **Small & hackable** – every component is contained in a handful of clear
  Python modules so that newcomers can read the code in an afternoon.
* **Block‑aware by design** – all core models expose their layers as a
  `ModuleList` called `blocks`, making it trivial to copy / swap / analyse
  individual parts of the network.
* **Extensible** – measurement units, early‑stop criteria, and optimiser
  engines are *plugins*: add a new class, import it in `Units.txt`, and it is
  available in the next run without touching the training loop.

The goal is to **experiment quickly with ideas like**

* analysing Hessian structure per parameter block,
* comparing synchronous vs. asynchronous block updates,
* measuring memory or communication‑time trade‑offs,
* trying exotic early‑stopping or curriculum schedules.

If you have similar curiosities, the framework should save you the boilerplate
work while staying transparent enough to be fully customisable.


---

# Directory Tour

```
├── src
│   ├── Architectures          # models & model‑loader helpers
│   ├── Buliding_Units         # measurement / stopper plugins & model frames
│   ├── Data                   # dataset download & synthetic data
│   ├── Optimizer_Engines      # different training loops
│   └── utils.py               # shared utilities
├── dol1.py                    # single‑run example script
├── master.py                  # multi‑GPU batch launcher
├── figures/                   # auto‑generated plots (Hessian heat‑maps …)
├── measurements/              # text logs from measurement units
└── reports/                   # high‑level run logs (hyper‑params, progress)
```

Below is a *functional* description of every important component.  For the
complete call signatures please consult the source – files are short and
documented inline.


## 1. `src/utils.py`

| Symbol | Purpose |
| ------ | ------- |
| `Reporter` | Simple run logger that writes a hash‑named file in `reports/` and records arbitrary messages plus program exit status. |
| `get_max_batch_size` | Heuristic GPU‑memory probe to pick the largest batch that fits. |
| `generate_data` | Thin wrapper around *torchvision* downloads so that all scripts share the same transforms. |
| `copy_block` / `copy_model` | Utilities that copy parameters between two *block‑compatible* models – crucial for the distributed engines. |
| `parse_arguments` | Automatically converts a default hyper‑parameter dict into a command‑line interface. |
| `parse_measurement_units` | Reads `measurements/Units.txt` and instantiates the listed measurement plugins. |


## 2. `src/Architectures/`

### feedforward_nn.py

* **FeedForwardNetwork** – fully‑connected network where *every linear layer is
  its own block* (`InitialBlock`, multiple `LinearBlock`s, `FinalBlock`).
* **FeedForwardCNN** – same spirit but with convolutional blocks.
* **EnsembleFeedForward{Network,CNN}** – hold *N* sub‑models and combine their
  predictions by averaging or by random voting.  All sub‑models’ blocks are
  concatenated into a global `ModuleList` so that engines / measurement units
  do not need to know about the ensemble internals.

### resnets.py

Re‑implementation of ResNet‑18/34 that preserves the layer list as blocks while
remaining weight‑compatible with the official torchvision variant.

### Models.py

Collection of *loader* helpers (`load_resnet18`, `load_feedforward_ensemble`, …)
so that experiment scripts can stay model‑agnostic.


## 3. `src/Buliding_Units/`

### MeasurementUnit.py (plugin interface)

* **MeasurementUnit (ABC)** – opens its own log file in `measurements/`, offers
  `measure(frame)` + `log_measurement()`.
* **Working_memory_usage** – sums parameter + optimiser state sizes.
* **Hessian_measurement** – brute‑force full Hessian via finite differences
  (usable on tiny models).
* **HessianBlockInteractionMeasurement** – **fast** power‑iteration routine
  that estimates the largest singular value of every off‑diagonal Hessian block
  without ever materialising the full matrix; writes a heat‑map to
  `figures/Measurements/Hessian_Measurement/`.

### StopperUnit.py (plugin interface)

Patience‑based (`EarlyStoppingStopper`) and threshold‑based
(`TargetStopper`) early‑stopping criteria specialised for loss or accuracy.

### Model_frames.py

`Frame` is the *runtime container* that owns the centre model, data loaders,
criterion, reporter, and registered plugins.  Two hierarchies extend it:

* **Disributed_frame** – maintains one model copy per parameter block plus
  optimisation & synchronisation helpers.
* **ImageClassifier_frame_blockwise** / **Regression_frame_blockwis** – task‑
  specific subclasses for block‑wise experiments.
* **ImageClassifier_frame_entire** – classic single‑model training.

The factory `generate_ModelFrame(H)` inspects the hyper‑parameter dict and
returns the right combination of model + frame.


## 4. `src/Optimizer_Engines/`

### BCD_engine.py

Includes three concrete training loops:

1. `train_blockwise_distributed` – spawns a process per block and alternates
   *K* local optimisation steps with a communication stage.
2. `train_blockwise_sequential` – executes the same algorithm sequentially
   (optionally on a random subset of blocks each round).
3. `train_entire` – vanilla epoch training for the non‑distributed baseline.

All loops share `log_progress()` which evaluates the model on a “big” batch and
records loss, accuracy, and timing information.


## 5. `src/Data/`

`generate_imagedata` downloads MNIST, CIFAR‑10/100, SVHN or a folder‑based
ImageNet subset and applies a uniform 32×32 transform.  `generate_regressiondata`
creates a trivial all‑ones regression toy set.


---

# How to Run an Experiment

```bash
# (1) install deps
python -m pip install -r requirements.txt

# (2) define / override hyper‑parameters on the cmd line
#     – every key in the default dict in dol1.py becomes a flag
python dol1.py --training_mode blockwise --model cnn_ensemble \
              --config "[[16,16],[32,32],[32,64]]" --rounds 2000

# (3) check results
cat reports/R*.txt                      # high‑level progress log
ls figures/Measurements/*/*.pdf         # Hessian heat‑maps, etc.
```

For large parameter sweeps you can use `master.py`, which launches many
`dol1.py` workers across all available GPUs and keeps track of progress in
`progress_tracker.json`.


---

# Extending the Framework

* **Add a new measurement** – create a subclass of `MeasurementUnit`, put it in
  `src/Buliding_Units/MeasurementUnit.py`, list its class name in
  `measurements/Units.txt`.  Next run of `dol1.py` will automatically import
  and execute it.
* **Add a new early‑stop criterion** – derive from `StopperUnit` or
  `EarlyStoppingStopper`.
* **Try a different optimisation algorithm** – add a new function to
  `src/Optimizer_Engines/` that accepts a `Frame` instance.  Because all models
  expose `frame.center_model` and optional `frame.distributed_models`, the loop
  does not need to know the architecture in advance.


---

# Acknowledgements & Licence

The code base was written for internal experimentation and is released under a
permissive licence (see `LICENSE`, if present).  Feel free to use, modify, and
build upon it – a citation or link back is always appreciated.
