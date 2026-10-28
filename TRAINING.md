# Training from scratch

The training sequence is **Cartesian parent -> road parent -> Full -> prediction**. Both parents start from random initialization; pretrained downloads are optional. The parent architectures, objectives, and example hyperparameters come from the models used by Full. Full freezes the trained parents and learns its additional components.

The full dataset and prepared road geometry will be released separately. Until then, the repository contains training code and three target-only examples, rather than the inputs needed for a real-data run. The manifest tool indexes already prepared windows; it does not convert raw trajectories into model inputs.

## 1. Install and prepare inputs

Install a compatible CANN/PyTorch/`torch_npu` stack and this package as described in the [README](README.md#ascend-installation). Run all commands below from the repository root. Training uses one Ascend NPU; choose it with `--device-id` (default `0`).

Expected layout:

```text
data/
  road_geometry_v10/       # metadata.json and matching numerical map arrays
  parents/train/*.npz      # prepared parent TRAIN windows
  parents/val/*.npz        # prepared parent validation windows
  full/train/*.npz         # prepared Full TRAIN windows
  full/val/*.npz           # prepared Full validation windows
  inference/*.npz          # observations and map inputs, without future labels
```

Each NPZ contains one unbatched window. Use the numerical array contract in [INTERFACE.md](INTERFACE.md#prepared-input-manifest). History has 41 positions and future labels have 20 steps at 15 s. Required arrays are:

| Stage | Observation inputs | Loss/validation labels |
|---|---|---|
| Cartesian parent | `history_xyz`, `history_xy_valid`, `history_z_valid`, `origin_context`, `time_features` | `target_xyz`, `target_xy_valid`, `target_z_valid` |
| Road parent | All 26 keys in `residual.OBSERVATION_KEYS` | The three XYZ label arrays plus `target_road_id`, `target_map_confidence`, **`target_road_s`** |
| Full | All 26 observation/map keys | The three XYZ label arrays plus `target_road_id`, `target_map_confidence` |
| Prediction | All 26 observation/map keys | None |

`target_xyz` is relative XYZ in metres with shape `[20, 3]`; validity masks have shape `[20]`. `target_road_id` is the map's internal road index (negative when unmatched). `target_road_s` is the matched position along the map's stored road coordinate in metres, with shape `[20]`. `target_map_confidence` has shape `[20]` and values in `[0, 1]`. Labels are used after forwarding observation inputs; they never enter route construction or the model's input list.

Both parents may read the same prepared parent windows; the Cartesian loader ignores the road fields. Parent and Full cohorts remain separate: the recorded parent runs used 4,157 TRAIN / 478 validation windows, while the later Full cohort used 4,061 / 461. Do not substitute one cohort's metrics for the other's. Vehicle/time split isolation must be established during preprocessing; the loader additionally rejects overlapping TRAIN/validation IDs.

Build manifests with hashes of the prepared files:

```bash
python tools/build_manifest.py --windows data/parents/train --split train --output data/parents/train/manifest.json
python tools/build_manifest.py --windows data/parents/val --split val --output data/parents/val/manifest.json
python tools/build_manifest.py --windows data/full/train --split train --output data/full/train/manifest.json
python tools/build_manifest.py --windows data/full/val --split val --output data/full/val/manifest.json
python tools/build_manifest.py --windows data/inference --split inference --output data/inference/manifest.json
```

Filenames become sample IDs. Keep them unique across TRAIN and validation, and retain the same IDs for the same source windows. Map files and prepared data are excluded from Git.

## 2. Train the two parents

Edit the paths in the two example configurations if needed, then run sequentially:

```bash
python -m roadtoken3d.pretrain cartesian --config configs/cartesian-parent.example.json --output runs/parents/cartesian
python -m roadtoken3d.pretrain road --config configs/road-parent.example.json --output runs/parents/road
```

Both recipes use seed 17, width 128, 800 AdamW updates, global batch 256, microbatch 16, learning rate `3e-4`, weight decay `0.01`, gradient clipping at 5, BF16 forward autocast, and Z loss weight 1. Their original warmup/cosine schedules are retained. The road parent uses travel-time graph encoding, 8 beams, a 48-road proposal limit, endpoint proposals, fixed-route inclusion, the initial-heading guard, and the hybrid ranking teacher. Full raises the proposal limit to 128 when loading this parent.

The standalone runner accumulates 16 microbatches on one NPU. The recorded parent runs used two NPUs with 8 microbatches per device for the same global batch. The new runner preserves the model and loss definitions, but sampling order and floating-point reduction order differ from that distributed execution; it does not promise identical historical weights or paper scores.

Validation runs every 100 updates. Selection keeps the first strict minimum of top-1 XY ADE over fully labeled 300 s validation windows. For the road parent, rejected windows receive the recorded 5,000 m selection penalty; accepted-only ADE and rejection counts are also logged. No test manifest is consumed.

Each run writes:

- `best.pt`: selected parent weights and construction options, ready for Full.
- `step-0100.pt`, ..., `step-0800.pt`: complete resume states including optimizer, draw stream, RNG, and the best weights so far.
- `protocol.json`, `progress.jsonl`, `selection.json`: source/data/config hashes, losses, validation, and selected step.

Resume into a new directory with the same configuration, source, inputs, map, and runtime:

```bash
python -m roadtoken3d.pretrain cartesian --config configs/cartesian-parent.example.json --resume runs/parents/cartesian/step-0400.pt --output runs/parents/cartesian-resumed
```

Use a numbered checkpoint for `--resume`; `best.pt` is a weights export, not a complete optimizer resume state. If resuming into another directory, point Full to that run's `best.pt`.

## 3. Train Full

[configs/current.example.json](configs/current.example.json) already references the two parent outputs above. Use the same road geometry as the road-parent run. The loader checks parent kind, model state, and recorded map hashes.

```bash
python -m roadtoken3d.ascend train --config configs/current.example.json --output runs/full
```

Full uses 1,600 updates, global batch 128, microbatch 16, FP32 AdamW, and the eight original pooled loss terms. Its default selection is fixed step 1,600. Its checkpoints retain both frozen parents, so the Full state can be reloaded strictly.

To resume Full:

```bash
python -m roadtoken3d.ascend train --config configs/current.example.json --resume runs/full/step-0050.pt --output runs/full-resumed
```

## 4. Predict

Set `inference_manifest` in the Full configuration to observation-only windows, then run:

```bash
python -m roadtoken3d.ascend predict --config configs/current.example.json --weights runs/full/step-1600.pt --output runs/prediction
```

Use the resumed run's checkpoint path if applicable. Outputs are written to `predictions.json` with sample ID, relative XYZ prediction, validity, and selected mode. Parent exports are still required to construct the model, and all weights must come from a trusted source.

## Verification scope

The earlier ARM/Ascend generated-data checks covered the existing Full components and checkpoint arithmetic. The newly added parent runner has received source, syntax, configuration, and packaging review only; it has not been executed on a device. No new training or device run was performed for this addition.
