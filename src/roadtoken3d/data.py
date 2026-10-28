"""Prepared observation-only inputs; future labels stay outside model.forward."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .residual import OBSERVATION_KEYS

LABEL_KEYS = ('target_xyz', 'target_xy_valid', 'target_z_valid',
              'target_road_id', 'target_map_confidence')

class WindowDataset(Dataset):
    def __init__(self, manifest, *, labels=False, observation_keys=OBSERVATION_KEYS, label_keys=LABEL_KEYS):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text())
        self.rows = self.manifest['windows']
        self.labels = labels
        self.keys = tuple(observation_keys) + (tuple(label_keys) if labels else ())
        if not self.rows or len({r['sample_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Nonempty unique sample IDs required')
        self.manifest_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        for row in self.rows:
            p = self.path.parent / row['file']
            if hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Prepared input hash differs: ' + row['sample_id'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        keys = self.keys
        with np.load(self.path.parent / row['file'], allow_pickle=False) as src:
            result = {k: torch.from_numpy(np.array(src[k], copy=True)) for k in keys}
        result['sample_id'] = row['sample_id']
        return result

class DrawStream:
    """Serializable training-only draw order; no dropped epoch tail."""
    def __init__(self, size, seed=17):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator).tolist()
        self.cursor = 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator).tolist()
                self.cursor = 0
            n = min(count - len(result), self.size - self.cursor)
            result.extend(self.order[self.cursor:self.cursor+n])
            self.cursor += n
        return result

    def state_dict(self):
        return dict(size=self.size, order=self.order, cursor=self.cursor,
                    generator=self.generator.get_state())

    def load_state_dict(self, state):
        if state['size'] != self.size or sorted(state['order']) != list(range(self.size)):
            raise ValueError('Draw stream dataset/order mismatch')
        if not 0 <= state['cursor'] <= self.size:
            raise ValueError('Invalid draw cursor')
        self.order, self.cursor = state['order'], state['cursor']
        self.generator.set_state(state['generator'])
