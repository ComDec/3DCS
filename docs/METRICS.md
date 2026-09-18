# Metrics

3DCS evaluates a molecular representation along three axes. Each evaluator supports two metric
versions:

- `--metric-version paper` (default): the definitions used for the published tables, so that those
  tables can be regenerated.
- `--metric-version v2`: an alternative set of definitions (full-precision inputs, separate kernel
  bandwidths, neighbour and population choices), described metric by metric on each page.

| Axis | Dataset (`EscheWang/3dcs` config) | Paper tables | Definitions |
|---|---|---|---|
| Geometry | `rotation` | Table 1 | [metrics/geometry.md](metrics/geometry.md) |
| Chirality | `chirality` | Tables 2, 4 | [metrics/chirality.md](metrics/chirality.md) |
| Energy | `traj_energies` (+ `traj_frames` for embedding generation) | Tables 3, 6, 7 | [metrics/energy.md](metrics/energy.md) |

Common conventions:

- Representation distances `Δ` are computed within one molecule (all conformers of a rotation
  molecule, all stereoisomer conformers of a chirality parent, or one trajectory window). RDKit bit
  vectors always use the Tanimoto distance.
- The chirality evaluator uses the Euclidean distance by default (`--distance euclidean`), which is
  what the published Table 2 used; `--distance cosine` is available. The geometry evaluator reports
  both cosine and Euclidean spaces (`--metrics`); Table 1 uses cosine. The energy evaluator uses the
  cosine distance.
- Scores are averaged per molecule (or per trajectory window) and then over molecules; non-finite
  per-molecule values are excluded from the mean. `summary.csv` reports the number of finite values.

The reproduction scripts and the reference values for each table are in [../reproduce/](../reproduce/README.md).
