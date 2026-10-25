# Three recorded trajectory examples

| File | Description |
|---|---|
| `sample_01.csv` | Long-range continuation |
| `sample_02.csv` | Recorded height variation; the manuscript case has mixed horizontal/vertical performance |
| `sample_03.csv` | Longitudinal overshoot case; retained to include a horizontal failure |

Each CSV contains the unchanged recorded future target for one of the manuscript's previously selected cases: 20 points at 15-second intervals, up to 300 seconds. XYZ values are metres relative to the final observed position. Original vehicle identifiers, sample hashes, absolute origins and timestamps are omitted. Values are copied from the existing six-decimal export without geometric modification.

Only target trajectories are supplied: no predicted trajectories, observed-history sequence, road graph, parent weights or full evaluation collection is included. These files illustrate the data columns and trajectory shapes; they are not complete trainable inputs or a representative statistical sample. No new sample search or model evaluation was performed.

The exact extraction is locally traceable. `index.json` provides release file hashes, descriptions and schema. Run `python tools/inspect_samples.py` to verify all three files using only the Python standard library.
