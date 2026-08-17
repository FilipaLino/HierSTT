# Pretrained models

All checkpoints from the paper are released on Google Drive.

**➜ [Download folder](PASTE_YOUR_GOOGLE_DRIVE_FOLDER_LINK_HERE)**

36 checkpoints: **6 seeds × 6 coherence weights**, one per cell of the α
ablation. Each is ≈18 MB (≈4.4 M parameters); the full set is ≈640 MB.

| | |
|---|---|
| Seeds | `1`, `7`, `21`, `42`, `123`, `2026` |
| α values | `0.0`, `0.3`, `0.5`, `0.7`, `0.99`, `0.999` |
| Main result | `seed = 42`, `α = 0.3` |
| Format | PyTorch `state_dict` (`torch.save(model.state_dict(), ...)`) |
| Naming | `model_seed_{seed}_alpha_{alpha}.pth`, with `.` → `_` in α |

Examples: `model_seed_42_alpha_0_3.pth`, `model_seed_2026_alpha_0_999.pth`,
`model_seed_1_alpha_0_0.pth`.

`α = 0` is the ablation with the coherence term switched off entirely — useful as
the "accuracy only" reference point.

## Downloading

```bash
pip install gdown

# everything (~640 MB)
python scripts/download_checkpoints.py --url "<folder link>"

# one cell of the grid
python scripts/download_checkpoints.py --url "<folder link>" --seed 42 --alpha 0.3

# one seed, every alpha
python scripts/download_checkpoints.py --url "<folder link>" --seed 42
```

Files land in `checkpoint/` by default (`--dest` to change). The script skips
files already present unless `--force` is given, and verifies checksums against
`checkpoint_manifest.csv` when that file is available.

You can also just download the folder from the browser and unzip it into
`checkpoint/`.

## Loading a checkpoint

The weights are a plain `state_dict`, so the model has to be constructed with the
same shapes before loading. Those shapes come from the data — the number of
hospitals and regions, and the per-level feature counts — so a checkpoint only
loads against a dataset with the schema in [`DATA.md`](DATA.md).

```python
import torch
from common.hierarchical_model import HierSTT

# config, input_sizes and outputs are built in run.py from the prepared data
model = HierSTT(config,
                r_input_size=input_sizes[0],
                h_input_size=input_sizes[1],
                d_model=128, nhead=4, num_layers=2,
                output_size=outputs, dropout_rate=0.2)

state = torch.load("checkpoint/model_seed_42_alpha_0_3.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

The simplest route is to let `run.py` do it:

```bash
python run.py --test --data /path/to/dataset.csv \
              --checkpoint checkpoint/model_seed_42_alpha_0_3.pth \
              --alpha 0.3
```

Pass the same `--alpha` the checkpoint was trained with — it does not change the
weights, but it does change the reported test loss.

### Expected shapes

| Component | Shape driver |
|---|---|
| `hospital_to_region` buffer | 81 entries, values 0–4 |
| Regional encoder input | `r_input_size / 5` features per region |
| Hospital encoder input | `h_input_size / 81` features per hospital |
| Decoder output | 28 days per entity |

A `size mismatch` on load almost always means the input CSV has a different
column set or a different hierarchy, not a corrupted file.

## Reference output

Evaluating the main checkpoint should print something close to this. Use it to
confirm your environment and data schema are set up correctly:

```
$ python run.py --test --data /path/to/dataset.csv \
                --checkpoint checkpoint/model_seed_42_alpha_0_3.pth --alpha 0.3

Test Evaluation
Test Loss (hierarchical): 0.2828
  National  MAE    797.34  RMSE    948.54  WAPE   4.83%
  Regional  MAE    167.49  RMSE    238.58  WAPE   5.08%
  Hospital  MAE     21.84  RMSE     33.37  WAPE  10.72%
  HAgE (prediction-side / vs ground truth)
    hospital -> regional    1.72%  /    5.60%
    hospital -> national    2.79%  /    4.66%
    regional -> national    2.98%  /    4.26%
```

Small deviations across hardware and library versions are expected. Large ones
(or a `size mismatch` on load) mean the input schema differs from the one in
[`DATA.md`](DATA.md).

## Training configuration

Every checkpoint was produced by the same recipe, varying only seed and α:

| | |
|---|---|
| Encoder window / horizon | 42 days / 28 days |
| National level | TFT — hidden 128, 2 LSTM layers, 4 heads, embedding dim 8, dropout 0.1 |
| Regional & hospital | Spatio-temporal Transformer — L = 2, d_model = 128, 4 heads, dropout 0.2 |
| Optimiser | AdamW, weight decay 1e-4 |
| Schedule | OneCycleLR, max_lr 3e-4, pct_start 0.05, cosine, div 25, final div 1e4 |
| Gradient clipping | max-norm 1.0 |
| Epochs | up to 200, early stopping patience 50 on validation loss |
| Selection | best validation loss |
| Target | `log1p`, `RobustScaler` fitted on train only |

Reproduce any single checkpoint with:

```bash
python run.py --train --data /path/to/dataset.csv --seed 42 --alpha 0.3
```

Determinism is requested via `torch.use_deterministic_algorithms(True,
warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, but exact bitwise
reproduction still depends on the GPU, driver and PyTorch version. Expect metrics
to match closely, not exactly, on different hardware.

## Verifying downloads

`checkpoint_manifest.csv` in the Drive folder lists every file with its size and
SHA-256. To check what you have:

```bash
python scripts/download_checkpoints.py --verify-only --dest checkpoint/
```

To regenerate the manifest after adding files:

```bash
python scripts/make_checkpoint_manifest.py --dir checkpoint/ \
                                           --out checkpoint_manifest.csv
```

## Licence and citation

The weights are released under the same MIT terms as the code. If you use them,
please cite the paper — see the [README](../README.md).
