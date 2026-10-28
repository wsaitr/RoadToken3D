"""Build a hash-bound manifest for prepared NPZ windows; data are not bundled."""
from pathlib import Path
import argparse
import hashlib
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val', 'dev', 'inference'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing manifest')
    files = sorted(args.windows.resolve().glob('*.npz'))
    if not files:
        parser.error('No prepared NPZ windows found')
    rows = []
    for path in files:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(b)
        rows.append(dict(sample_id=path.stem,
                         file=os.path.relpath(path, args.output.parent.resolve()), sha256=h.hexdigest()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(split=args.split, windows=rows), indent=2) + '\n')
    print(json.dumps(dict(manifest=str(args.output), windows=len(rows), split=args.split)))


if __name__ == '__main__':
    main()
