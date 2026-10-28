# Model interfaces, shared core, and Ascend execution

## Model and checkpoint identity

The default is `joint-timegate6-anchor`, the current manuscript's Full model. The other five names in `current_model.ARMS_V6` preserve the matching control configurations. Parent checkpoints supply their original model options, weights and immutable geometry; a trained Full state can then be loaded with strict parameter matching.

The old endpoint-only source bundle is superseded and is not part of this upload. `current_model.py` assembles the learned occupancy and height gates used by the current model. Imports and module binding are made local; no former training controller is imported and no global backend monkeypatch is installed.

Model definitions, training objectives, prepared-data handling, and pooled updates form one shared PyTorch core. The Ascend wrapper supplies device initialization, checkpoint orchestration, and device RNG handling. Resume requires the same Ascend source/data/config protocol, including optimizer, draw order and RNG states. Loading a model's weights is distinct from resuming another device's optimizer trajectory. No cross-device bitwise or historical draw-order equivalence is claimed.

The published Full outputs used fixed1600 weights. The example configuration therefore declares `fixed-final`. `validation-first-strict-min` is an explicit alternative for a separately declared run; the resulting outputs must not be mixed or relabeled. Validation is evaluated every200 steps, checkpoints saved every50, with1600 updates/global128/micro16 and fixed FP32 AdamW3e-4/weight decay1e-3.

## Prepared input manifest

The three example CSVs are qualitative target-only previews. To run the model, prepare the 26 observation/map arrays in `residual.OBSERVATION_KEYS` for each window, using the original history-only preprocessing and strict XYZ connectivity protocol. Training additionally needs `target_xyz`, `target_xy_valid`, `target_z_valid`, `target_road_id` and `target_map_confidence`. Do not pass these labels to the model's forward method.

Each NPZ holds one unbatched window and uses ordinary numerical arrays; object/pickle arrays are rejected. A separate JSON manifest contains:

```json
{
  "split": "train",
  "windows": [
    {"sample_id": "example-window-id", "file": "window_0001.npz", "sha256": "REPLACE_WITH_ACTUAL_FILE_SHA256"}
  ]
}
```

Use `val` or `dev` for the validation manifest. Prepared file hashes are checked before use. TRAIN and validation sample IDs must not overlap; upstream vehicle/time isolation and source validity also need to be established by the dataset owner. Predict mode loads only observation arrays and does not compute a hindsight teacher.

## Ascend adaptation

- Device setup requires `torch_npu` and actual NPU availability; there is no training/forecast CPU fallback.
- Model, geometry, limits, teacher rules, losses and gate supervision follow the current v6 code.
- Eight numerator/denominator terms are pooled before dividing. Count/replay verifies values, weights and RNG before updates; frozen parents are checked.
- The prior device's FP64 matrix prefix is not assumed available on Ascend. A conservative on-device sequential FP32/int32 scan preserves exact zero-padded tails and differentiation; output dtype is retained. Its speed is not qualified. CPU reference code uses the standard prefix operation.
- A single-device generated-data check passed on the recorded stack: real updates, raw NPU Adam checkpoint save/load, fresh-process state replay, prefix gradients and observation-only forwarding. Full-data execution, throughput and production-scale memory still require qualification before scientific use.

Preparing or uploading this repository starts no paid device, real training or evaluation. The synthetic check uses a generated tiny road graph and contains zero real samples and three tiny optimizer updates. It tests nonzero gates, all eight losses, future-label exclusion, gradients, pooled updates, input hashes, exact model/Adam/draw/RNG reload and prefix behavior; it cannot validate a full NPU training run.

## Model metadata

Output metadata describes model identity, geometry, masks, and coordinate semantics. TRAIN-only teacher checks, observation-only prediction inputs, finite-value guards, and source/data/checkpoint integrity checks remain active. Training commands and their required input files are documented in the README.

## Runnable demonstration and paper reproduction

`tests/smoke_current.py` is the existing self-contained generated-data demonstration. It builds the actual model classes with tiny dimensions and initialized parents, performs updates, and exercises checkpoint resume without private assets. The three recorded CSV cases are target-only illustrations. Full manuscript reproduction additionally requires the full prepared dataset, map geometry, and frozen trained parent weights; these are not supplied in this minimal release.
