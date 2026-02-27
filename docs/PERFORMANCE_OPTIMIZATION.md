# Performance Notes

This document summarizes performance considerations for large-scale evaluations.

## Chirality

- Distance matrices can be large. Consider limiting `unsup_kmax` if needed.
- Unsupervised metrics are the most expensive; skip them only if you do not need them.

## Trajectory

- Pairwise distances scale quadratically with the window size.
- Reduce `window` or `n_samples` for quick iterations.

## Rotation

- RMSD computations dominate runtime. Use smaller subsets while testing.
