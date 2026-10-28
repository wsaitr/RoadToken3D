"""Train the Cartesian or road parent before training the Full model."""
from pathlib import Path
from contextlib import nullcontext
import argparse
import json
import math
import os
import torch
from torch.utils.data import DataLoader, default_collate

from .ascend import setup, move, sha, source_pins, atomic_save, geometry_pins
from .data import WindowDataset, DrawStream, LABEL_KEYS
from .dense_model import DenseCoordinateForecaster
from .road_graph import RoadGraph
from .route_model_v13 import RoadNativeV13
from .residual import OBSERVATION_KEYS
from .parent_objectives import dense_loss, v13_loss
from .optimization import capture_rng, restore_rng

CARTESIAN_KEYS = OBSERVATION_KEYS[:5]
CARTESIAN_LABELS = ('target_xyz', 'target_xy_valid', 'target_z_valid')
ROAD_LABELS = LABEL_KEYS + ('target_road_s',)


def replace_save(path, value):
    """Replace a run-local export atomically; numbered checkpoints are immutable."""
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(value, temporary)
    os.replace(temporary, path)


def autocast(device, precision):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if precision == 'bf16' else nullcontext())


def learning_rate_factor(kind, index, settings):
    """The two recorded parent runners use different warmup/cosine schedules."""
    steps = settings['steps']
    if kind == 'road':
        warmup = min(settings['warmup_steps'], steps)
        warm = min(1., (index + 1) / warmup) if warmup else 1.
        return warm * (.05 + .95 * .5 * (1. + math.cos(math.pi * index / steps)))
    warmup = min(settings['warmup_steps'], max(0, steps - 1))
    if index < warmup:
        return (index + 1) / warmup
    return .1 + .9 * .5 * (1. + math.cos(
        math.pi * (index - warmup) / max(1, steps - warmup)))


def forward(model, batch, kind):
    keys = CARTESIAN_KEYS if kind == 'cartesian' else OBSERVATION_KEYS
    return model(*(batch[k] for k in keys))


@torch.no_grad()
def evaluate(model, dataset, device, kind, precision):
    """Use complete 300 s labels; keep the road parent's 5,000 m rejection cost."""
    model.eval()
    total = accepted_sum = 0.
    complete_count = accepted_count = incomplete_count = 0
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        b = move(batch, device)
        with autocast(device, precision):
            output = forward(model, b, kind)
        prediction = output['trajectories'].float()
        choice = output['mode_logits'].argmax(-1)
        rows = torch.arange(len(prediction), device=device)
        selected = prediction[rows, choice]
        accepted = (output['forecast_valid'].bool() if kind == 'road'
                    else torch.ones(len(prediction), dtype=torch.bool, device=device))
        complete = b['target_xy_valid'][:, :20].bool().all(-1)
        if not torch.isfinite(selected[accepted]).all():
            raise FloatingPointError('Nonfinite accepted parent forecast')
        error = torch.linalg.vector_norm(
            torch.where(accepted[:, None, None], selected[..., :2], 0.)
            - b['target_xyz'][:, :20, :2].float(), dim=-1).mean(-1)
        good = complete & accepted
        if not torch.isfinite(error[good]).all():
            raise FloatingPointError('Nonfinite labeled validation error')
        total += float(torch.where(accepted, error, 5000.)[complete].sum())
        accepted_sum += float(error[good].sum())
        complete_count += int(complete.sum())
        accepted_count += int(good.sum())
        incomplete_count += int((~complete).sum())
    if complete_count == 0:
        raise ValueError('Validation has no fully labeled 300 s window')
    return dict(selection_xy_ade300_m=total / complete_count,
                accepted_xy_ade300_m=accepted_sum / accepted_count if accepted_count else None,
                complete_windows=complete_count, accepted_windows=accepted_count,
                rejected_windows=complete_count - accepted_count,
                incomplete_windows=incomplete_count, rejection_penalty_m=5000. if kind == 'road' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('cartesian', 'road'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--resume', type=Path, help='A numbered parent checkpoint, not best.pt')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    settings = config['training']
    if config['kind'] != args.kind:
        parser.error('Parent kind differs from the configuration')
    if args.output.exists():
        parser.error('Use a new output directory, including when resuming')
    for name in ('steps', 'global_batch', 'micro_batch', 'validate_every', 'save_every'):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            parser.error(name + ' must be a positive integer')
    if settings['global_batch'] % settings['micro_batch']:
        parser.error('global_batch must be divisible by micro_batch')
    if (settings['precision'] not in ('fp32', 'bf16')
            or isinstance(settings['warmup_steps'], bool)
            or not isinstance(settings['warmup_steps'], int) or settings['warmup_steps'] < 0):
        parser.error('Invalid precision or warmup_steps')
    if any(not math.isfinite(settings[k]) for k in ('learning_rate', 'weight_decay', 'gradient_clip', 'z_weight')):
        parser.error('Nonfinite optimizer/loss setting')
    if min(settings['learning_rate'], settings['gradient_clip']) <= 0 or min(settings['weight_decay'], settings['z_weight']) < 0:
        parser.error('Invalid optimizer/loss setting')
    options = config['model_options']
    if options.get('future_steps') != 20 or options.get('history_steps') != 41:
        parser.error('The paper parent recipes require 41 history and 20 future steps')
    if options.get('interval_seconds', 15.) != 15.:
        parser.error('The parent objectives require 15 s forecast intervals')
    if args.kind == 'cartesian' and (options.get('modes') != 1 or options.get('architecture') != 'attention'):
        parser.error('Full requires the single-mode attention Cartesian parent')
    keys = CARTESIAN_KEYS if args.kind == 'cartesian' else OBSERVATION_KEYS
    labels = CARTESIAN_LABELS if args.kind == 'cartesian' else ROAD_LABELS
    train = WindowDataset(config['train_manifest'], labels=True, observation_keys=keys, label_keys=labels)
    val = WindowDataset(config['validation_manifest'], labels=True, observation_keys=keys, label_keys=labels)
    if train.manifest.get('split') != 'train' or val.manifest.get('split') not in ('val', 'dev'):
        parser.error('Explicit TRAIN and validation manifests are required')
    if {r['sample_id'] for r in train.rows} & {r['sample_id'] for r in val.rows}:
        parser.error('TRAIN and validation IDs overlap')
    device = setup(args.device_id, config.get('seed', 17))
    if args.kind == 'cartesian':
        model = DenseCoordinateForecaster(**options)
        geometry = None
    else:
        geometry = geometry_pins(config['geometry'])
        model = RoadNativeV13(RoadGraph(config['geometry']), **options)
        if model.proposer.delayed_messages is not None:
            raise ValueError('Pretrain the road parent before attaching Full-model delayed messages')
    model = model.to(device)
    import torch_npu
    protocol = dict(schema='roadtoken3d-parent-training-v1', parent_kind=args.kind,
                    model_options=options, config=config, config_sha256=sha(args.config),
                    source_sha256=source_pins(), geometry_sha256=geometry,
                    data_sha256=dict(train=train.manifest_sha256, validation=val.manifest_sha256),
                    device_type='npu', torch_version=str(torch.__version__),
                    torch_npu_version=str(torch_npu.__version__), precision=settings['precision'],
                    actual_train_samples=len(train), actual_val_samples=len(val),
                    effective_global_batch=settings['global_batch'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    stream = DrawStream(len(train), config.get('seed', 17))
    step = 0
    best = None
    best_model = None
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        required = {'protocol', 'model', 'optimizer', 'draw_stream', 'rng', 'step', 'best', 'best_model'}
        if not required.issubset(state) or state['protocol'] != protocol:
            raise ValueError('Resume requires a numbered checkpoint with the same parent protocol')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['draw_stream'])
        restore_rng(state['rng'], device)
        step, best, best_model = state['step'], state['best'], state['best_model']
        if not 0 <= step <= settings['steps']:
            raise ValueError('Resume step is outside the configured budget')
    args.output.mkdir(parents=True)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    def export_best():
        if best_model is not None:
            replace_save(args.output / 'best.pt', dict(protocol=protocol, model=best_model,
                         step=best['step'], score=best['score']))
    export_best()
    batches = settings['global_batch'] // settings['micro_batch']
    for step in range(step + 1, settings['steps'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = stream.take(settings['global_batch'])
        loss_sum = 0.
        for start in range(0, len(indices), settings['micro_batch']):
            batch = move(default_collate([train[i] for i in indices[start:start + settings['micro_batch']]]), device)
            with autocast(device, settings['precision']):
                output = forward(model, batch, args.kind)
            loss = (dense_loss(output, batch, model.output_xy_scale, settings['z_weight'])
                    if args.kind == 'cartesian' else
                    v13_loss(output, batch, settings['z_weight'], rank_teacher=config['rank_teacher']))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite parent training loss')
            (loss / batches).backward()
            loss_sum += float(loss.detach()) / batches
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        lr = settings['learning_rate'] * learning_rate_factor(args.kind, step - 1, settings)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite parent parameter after update')
        metrics = None
        if step % settings['validate_every'] == 0 or step == settings['steps']:
            prior_rng = capture_rng(device)
            metrics = evaluate(model, val, device, args.kind, settings['precision'])
            restore_rng(prior_rng, device)
            score = metrics['selection_xy_ade300_m']
            if best is None or score < best['score']:
                best = dict(step=step, score=score)
                best_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                export_best()
        if step % settings['save_every'] == 0 or step == settings['steps']:
            atomic_save(args.output / f'step-{step:04d}.pt', dict(protocol=protocol, step=step,
                        model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        optimizer=optimizer.state_dict(), rng=capture_rng(device),
                        draw_stream=stream.state_dict(), best=best, best_model=best_model))
        record = dict(step=step, loss=loss_sum, learning_rate=lr, gradient_norm=float(norm), validation=metrics)
        with (args.output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    (args.output / 'selection.json').write_text(json.dumps(dict(rule='first-strict-validation-minimum', best=best,
             export='best.pt', resume='numbered step-*.pt only'), indent=2) + '\n')


if __name__ == '__main__':
    main()
