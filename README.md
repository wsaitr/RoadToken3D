# RoadToken3D

RoadToken3D forecasts 3D vehicle trajectories from observed motion and road geometry. This repository contains the shared PyTorch model and training core used by the current manuscript, a small Ascend execution layer, and **three recorded trajectory examples**.

## Model

The released model is `joint-timegate6-anchor-20261023-v6`. Its components are:

- A learned Cartesian XYZ branch built on frozen parent models.
- Road-relative 3D decoding with shared longitudinal progress and bounded road corrections: 20 m horizontally and 2 m vertically.
- Travel-time delayed graph messages and earliest-arrival constraints.
- A learned time-occupancy gate with gain 10, a 1 m supervision deadband, and class-balanced supervision.
- A learned height anchor applied only to Z.
- Six trajectory/graph loss terms plus two gate losses, with numerator/denominator pooling over 128 training examples.

The model, losses, data handling, and pooled update arithmetic use standard PyTorch. Ascend-specific code is limited to device initialization, device RNG handling, and a padding-stable prefix-sum helper. Parent-module version names identify inherited dependencies; the top-level model is the current v6 implementation.

## Quick start: runnable synthetic demo

With NumPy and PyTorch installed, run the existing demonstration from the repository root:

```bash
python -m pip install -e . --no-deps
python tests/smoke_current.py
```

The demo generates a tiny road graph, complete observation/label tensors, and initialized parent modules. It exercises the same model and eight objectives, performs tiny parameter updates, and checks checkpoint resume. It needs no private files or accelerator. Its generated inputs and initialized weights are for demonstrating execution, not reproducing the manuscript's scores.

## Ascend installation

Prepare compatible CANN, PyTorch, and `torch_npu` versions using the [official Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md), then install this package:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -e . --no-deps
python -m roadtoken3d.ascend --help
```

## Training, prediction, and resume

The complete code path is **train the two parents -> train Full -> predict**. Pretrained parent downloads are optional: [TRAINING.md](TRAINING.md) documents training both parents from random initialization, prepared-data manifests, selection, and checkpoint resume. The full dataset and road geometry will be released separately.

The example configurations reference the parent outputs and prepared input paths in that guide. See [INTERFACE.md](INTERFACE.md) for input arrays, split checks, and checkpoint compatibility.

```bash
# Train the two parents on an Ascend NPU.
python -m roadtoken3d.pretrain cartesian --config configs/cartesian-parent.example.json --output runs/parents/cartesian
python -m roadtoken3d.pretrain road --config configs/road-parent.example.json --output runs/parents/road

# Train Full using the selected parents.
python -m roadtoken3d.ascend train --config configs/current.example.json --output runs/full

# Resume a checkpoint with the same source, configuration, data, and runtime.
python -m roadtoken3d.ascend train --config configs/current.example.json --resume runs/full/step-0050.pt --output runs/resumed

# Predict using trained weights and observation/map inputs.
python -m roadtoken3d.ascend predict --config configs/current.example.json --weights runs/full/step-1600.pt --output runs/prediction
```

Use the actual checkpoint filename produced by your run for `--resume`. Training and prediction require an Ascend NPU; the tensor core also supports CPU implementation checks. The Full configuration uses 41 history positions, 20 forecast steps at 15 s, 1,600 updates, a global batch of 128, microbatches of 16, and FP32 AdamW. Each parent recipe uses 800 updates, global batch 256, and BF16 forward autocast.

## Examples and checks

Inspect the three supplied trajectories without installing model dependencies:

```bash
python tools/inspect_samples.py
```

The examples contain relative XYZ **future targets only**, covering long-range continuation, height variation, and a horizontal failure case. They do not include vehicle identifiers, absolute locations, observation histories, road graphs, predictions, or parent weights. They illustrate the trajectory format and are not complete training inputs. Full-data release is planned after paper acceptance.

## Validation

Small generated-data checks passed on native ARM CPU and a single Ascend device on 2026-10-25. The Ascend runtime reported `Ascend910B4` with PyTorch 2.3.1 and `torch_npu` 2.3.1.post2.dev20241029. Three optimizer updates covered a continuous run and a fresh-process checkpoint resume, with matching model, Adam, sampling, and RNG states. Prefix operations, gradients, Z-only anchoring, and future-label exclusion were also checked.

These earlier checks cover small-scale Full-model execution. The newly added parent runner has received static source/configuration review only and has not been run on hardware. Full-data accuracy, throughput, production-scale memory, and cross-device numerical equivalence require separate evaluation.

## Code map

| Path | Purpose |
|---|---|
| `src/roadtoken3d/current_model.py` | Shared v6 model, occupancy gate, height anchor, and control configurations |
| `src/roadtoken3d/current_loss.py`, `optimization.py` | Shared objectives, gate supervision, and pooled updates |
| `src/roadtoken3d/residual.py`, `endpoint.py` | Road-relative displacement and endpoint-stable decoding |
| `src/roadtoken3d/data.py` | Prepared inputs, hashes, and reproducible sampling |
| `src/roadtoken3d/pretrain.py`, `parent_objectives.py` | Cartesian/road parent training, original losses, selection, and resume |
| `src/roadtoken3d/ascend.py`, `prefix.py` | Ascend execution and prefix-operation compatibility |
| `tools/build_manifest.py`, `configs/`, `TRAINING.md` | Prepared-file indexing, training recipes, and complete workflow |
| Other model modules | Direct dependencies of the frozen parent models |
| `examples/` | Three recorded target trajectories and their hashes |
