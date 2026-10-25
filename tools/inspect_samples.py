"""Validate the three saved trajectory examples without running a model."""
from pathlib import Path
import csv
import hashlib
import json
import math

root = Path(__file__).resolve().parents[1]
index = json.loads((root / "examples/index.json").read_text())
assert index["sample_count"] == len(index["samples"]) == 3
for item in index["samples"]:
    p = root / item["file"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
    with p.open(newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 20
    assert [int(float(row["t_sec"])) for row in records] == list(range(15, 301, 15))
    assert all(math.isfinite(float(row[k])) for row in records for k in ["x_m", "y_m", "z_m"])
    assert all(row["xy_valid"] == row["z_valid"] == "1" for row in records)
    print(f'{item["sample"]}: 20 valid XYZ targets, 15--300 s, SHA256 verified')
print("3 examples verified; no model execution.")
