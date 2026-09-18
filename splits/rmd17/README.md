# rMD17 train/test splits

These are the five official train/test index splits distributed with the revised MD17 dataset
(rMD17; Christensen & von Lilienfeld, 2020, <https://doi.org/10.6084/m9.figshare.12672038>).
The files are copied byte for byte from the rMD17 release (`splits/index_{train,test}_0{1..5}.csv`);
we did not modify them.

| File | Rows | Content |
|---|---|---|
| `index_train_0{1..5}.csv` | 1000 each | 0-based frame indices into each `rmd17_<molecule>.npz` |
| `index_test_0{1..5}.csv` | 1000 each | 0-based frame indices, disjoint from the matching train split |

The same indices apply to all ten molecules. In the 3DCS trajectory data (`EscheWang/3dcs`,
configs `traj_frames` / `traj_energies`), frame `frame_idx` of `rmd17_<molecule>` is row `frame_idx`
of the rMD17 `.npz`, so these indices can be used directly on the HF data.

## Use in the paper

- The zero-shot energy benchmark (Tables 3, 6 and 7) does **not** use these splits; it samples
  windows over all frames.
- The fine-tuning inputs for Tables 8 and 9 are split **01** (`index_train_01.csv` /
  `index_test_01.csv`, 1000/1000 frames, no validation split) for all ten molecules. Fine-tuning code
  and checkpoints for rMD17 are not part of this repository.

## SHA-256

```
67f2c51e7c00bbc1f9fc5ba90863e5d526e6bca0d885bb68f9a1df4a5b7825cb  index_train_01.csv
65bfd73039f32a73586b84d799b96cea4fb4809a642611a7e8e8d94b9a5982f8  index_test_01.csv
437d61f05031ae35ff3bcf6aa22b2de4e56f9cfa10af169a5c604871b11056fc  index_train_02.csv
161f85238146e9398f84b22d250c350cb51916af5f93a51b0b8e0abb671883d8  index_test_02.csv
9d87806d98b902298568844b3a3bf694de51c2266066f2d7cd90810bcf4e2524  index_train_03.csv
2d5dbebdf942ead1b44fa496e037a6a6c4814d1615907e5f42ca2f354dca3925  index_test_03.csv
8d4d44298dbb15895472b7da8c0429c14449e88390450ecc202822d50a36b6e6  index_train_04.csv
b6716781813dfc569e820f45f9741d18b4c8fb1ac71a7a17b21a6780a9b37d24  index_test_04.csv
daa7e1351047d67a9a5d91ee44968ce708138e0f4f7a47fc38cc9c8d109fd636  index_train_05.csv
9cd07efd184dced6320b8656cf9efef2a59534d16eec85e536a2b04aa40775fe  index_test_05.csv
```

## License and citation

rMD17 is published on figshare under CC0; see the rMD17 record for its terms. If you use these
splits, please cite rMD17:

```bibtex
@article{christensen2020role,
  title   = {On the role of gradients for machine learning of molecular energies and forces},
  author  = {Christensen, Anders S. and von Lilienfeld, O. Anatole},
  journal = {Machine Learning: Science and Technology},
  volume  = {1},
  number  = {4},
  pages   = {045018},
  year    = {2020}
}
```
