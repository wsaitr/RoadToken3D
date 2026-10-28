"""Ascend train/predict entry point for the shared RoadToken3D PyTorch core.

Model definitions, objectives, input handling, and pooled updates are shared.
This module initializes the NPU, records the execution protocol, and provides
training, prediction, and compatible-checkpoint resume commands.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from .current_model import from_parents, ARMS_V6
from .dense_model import DenseCoordinateForecaster
from .route_model_v13 import RoadNativeV13
from .route_model_v10 import select_prediction
from .road_graph import RoadGraph
from .residual import OBSERVATION_KEYS
from .data import WindowDataset, DrawStream
from .optimization import loss_parts, pooled_arithmetic_step, capture_rng, restore_rng

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

def setup(device_id, seed):
    import torch_npu  # Registers the Ascend backend; no implicit CPU fallback.
    if not torch.npu.is_available():
        raise RuntimeError('An actual Ascend NPU with a compatible torch_npu/CANN stack is required')
    torch.npu.set_device(device_id)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    return torch.device(f'npu:{device_id}')

def build(config, device):
    # Load only trusted, author-controlled checkpoints.
    c = torch.load(config['cartesian_parent'], map_location='cpu', weights_only=False)
    r = torch.load(config['road_parent'], map_location='cpu', weights_only=False)
    direct = DenseCoordinateForecaster(**c['protocol']['model_options'])
    direct.load_state_dict(c['model'], strict=True)
    road = RoadNativeV13(RoadGraph(config['geometry']), **r['protocol']['model_options'])
    road.load_state_dict(r['model'], strict=True)
    road.proposer.maximum_roads = 128
    model = from_parents(direct, road, arm=config.get('arm', ARMS_V6[0]),
                         hops=48, message_width=32, checkpoint_chunk=8,
                         seed=config.get('seed',17))
    if model.future_steps != 20 or model.dt != 15.:
        raise ValueError('The current paper configuration requires 20 steps at 15 s')
    return model.to(device)

def source_pins():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def frozen_guard(model):
    names = set(model.frozen_road_parent_parameter_names)
    frozen = list(model.cartesian.parameters()) + [p for n,p in model.road.named_parameters() if n in names]
    if any(p.requires_grad or p.grad is not None for p in frozen):
        raise RuntimeError('A frozen parent was thawed or acquired a gradient')

@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    xy_sum = 0.
    count = 0
    saved = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        b = move(batch, device)
        out = model(*(b[k] for k in OBSERVATION_KEYS))
        selected = select_prediction(out)
        if selected['prediction'].device != device:
            raise RuntimeError('Explicit device fallback')
        pred = selected['prediction']
        saved.append(dict(sample_id=batch['sample_id'][0], prediction=pred[0].cpu().tolist(),
                          forecast_valid=bool(selected['forecast_valid'][0]),
                          selected_mode=int(selected['selected_mode'][0])))
        if dataset.labels:
            mask = b['target_xy_valid'][:, :20].bool()
            valid = mask.any(-1) & selected['forecast_valid']
            if not valid.all():
                raise ValueError('Do not silently drop rejected/labelless validation windows')
            d = torch.linalg.vector_norm(pred[..., :2] - b['target_xyz'][:,:20,:2], dim=-1)
            xy_sum += float(((d*mask).sum(-1)/mask.sum(-1).clamp_min(1)).sum())
            count += len(pred)
    return (xy_sum/count if count else None), saved

def atomic_save(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    if tmp.exists() or path.exists():
        raise FileExistsError(path)
    torch.save(value, tmp)
    os.replace(tmp, path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train','predict'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--weights', type=Path, help='Trusted model state for prediction or explicit warm start')
    parser.add_argument('--resume', type=Path, help='Checkpoint from this same Ascend runtime and protocol')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        parser.error('Use a new output directory; original evidence is never overwritten')
    if args.resume and args.weights:
        parser.error('--resume and --weights are mutually exclusive')
    if config.get('arm',ARMS_V6[0]) not in ARMS_V6:
        parser.error('Unsupported model configuration')
    device = setup(args.device_id, config.get('seed',17))
    model = build(config, device)
    protocol = dict(schema='roadtoken3d-current-ascend-v1', config=config,
                    source_sha256=source_pins(), config_sha256=sha(args.config),
                    parent_sha256={k:sha(config[k]) for k in ['cartesian_parent','road_parent']},
                    geometry_sha256={p.name:sha(p) for p in sorted(Path(config['geometry']).iterdir())
                        if p.is_file() and p.suffix in ['.npy','.json']},
                    device_type='npu', torch_version=torch.__version__, precision='FP32')
    if args.weights:
        w = torch.load(args.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(w.get('model',w), strict=True)
    args.output.mkdir(parents=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if args.mode == 'predict':
        if not args.weights or args.resume:
            parser.error('Prediction requires --weights and never consumes a training resume state')
        dataset = WindowDataset(config['inference_manifest'], labels=False)
        _, saved = evaluate(model,dataset,device)
        (args.output/'predictions.json').write_text(json.dumps(saved)+'\n')
        return
    train = WindowDataset(config['train_manifest'], labels=True)
    if train.manifest.get('split') != 'train':
        raise ValueError('Training requires an explicitly declared TRAIN manifest')
    val = WindowDataset(config['validation_manifest'], labels=True)
    if val.manifest.get('split') not in ['val','dev']:
        raise ValueError('Selection requires a distinct validation/development manifest')
    if {r['sample_id'] for r in train.rows} & {r['sample_id'] for r in val.rows}:
        raise ValueError('TRAIN and validation sample IDs overlap')
    protocol['data_sha256'] = {'train':train.manifest_sha256,'validation':val.manifest_sha256}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
    stream = DrawStream(len(train), config.get('seed',17))
    step = 0
    best = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['protocol'] != protocol:
            raise ValueError('Resume source/data/device protocol differs')
        model.load_state_dict(state['model'],strict=True)
        opt.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['draw_stream'])
        restore_rng(state['rng'],device)
        step, best = state['step'], state['best']
    selection = config.get('selection','fixed-final')
    if selection not in ['fixed-final','validation-first-strict-min']:
        raise ValueError('Declare the checkpoint selection rule explicitly')
    for step in range(step+1,1601):
        frozen_guard(model)
        indices = stream.take(128)
        batches = [move(default_collate([train[i] for i in indices[j:j+16]]),device)
                   for j in range(0,128,16)]
        cache = {}
        record = pooled_arithmetic_step(model,opt,batches,
            lambda m,b:loss_parts(m,b,cache),device)
        frozen_guard(model)
        score = None
        if step % 200 == 0:
            prior_rng = capture_rng(device)
            score,_ = evaluate(model,val,device)
            restore_rng(prior_rng,device)
            if best is None or score < best['xy_ade300']:
                best = dict(step=step,xy_ade300=score)
        if step % 50 == 0:
            atomic_save(args.output/f'step-{step:04d}.pt',dict(protocol=protocol,step=step,best=best,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=opt.state_dict(),rng=capture_rng(device),draw_stream=stream.state_dict()))
        with (args.output/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,xy_ade300=score,**record))+'\n')
    selected = 1600 if selection == 'fixed-final' else best['step']
    (args.output/'selection.json').write_text(json.dumps(dict(rule=selection,selected_step=selected,
        validation_best=best),indent=2)+'\n')

if __name__ == '__main__':
    main()
