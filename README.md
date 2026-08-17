# HierSTT: Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting

<p align="center">
  <img src="https://github.com/user-attachments/assets/13dc998e-8f21-40d5-b591-2d16ef30aed6"  width="70%" height="70%">
</p>

**Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting**
Filipa Lino, Bárbara Tavares, Carlos Santiago, Cláudia Soares, Manuel Marques
· ECML-PKDD 2026, SoGood Workshop · [arXiv:2607.27106](https://arxiv.org/abs/2607.27106)

---

Emergency Department planning needs forecasts at several levels at once:
hospitals need local demand for staffing, regions need it to coordinate units,
and national authorities need system-wide projections. Forecasting each level
separately produces **incoherent** predictions — hospital forecasts that do not
sum to the regional forecast, regional forecasts that do not sum to the national
one.

HierSTT predicts all levels jointly in a single end-to-end model. A Temporal
Fusion Transformer forecasts national demand; spatio-temporal Transformer
encoder–decoders forecast regional and hospital demand, each conditioned on the
level above. A coherence-aware loss penalises cross-level inconsistency during
training rather than reconciling forecasts afterwards.

On a nationwide Portuguese dataset (81 hospitals, 5 regions, 3 years), this
reduces average WAPE by about 32% relative to the best non-hierarchical deep
learning baseline while producing near-coherent forecasts across levels.

<p align="center">
  <em>national forecast → conditions regional decoding → conditions hospital decoding</em><br>
  42-day encoder window → 28-day horizon
</p>

## Quick start

```bash
git clone https://github.com/FilipaLino/HierSTT.git
cd HierSTT
pip install -r requirements.txt

# Train HierSTT end to end vwith the published configuration
python scripts/run.py --train --data /path/to/dataset.csv --seed 42 --alpha 0.3

# evaluate a checkpoint on the held-out test split
python scripts/run.py --test --data /path/to/dataset.csv --checkpoint checkpoint/model_seed_42_alpha_0_3.pth
```

Both modes print per-level MAE, RMSE and WAPE, plus HAgE for the three
aggregation transitions. The dataset is **not published at this time** while its redistribution terms are
being confirmed. It was assembled from public Portuguese sources; every source
is listed in [`data/DATA.md`](data/DATA.md), together with the full column schema,
so an equivalent table can be rebuilt from them. If and when redistribution is
cleared, a download link will be added here.

All 36 checkpoints from the paper — **seeds `1, 7, 21, 42, 123, 2026` × α values
`0.0, 0.3, 0.5, 0.7, 0.99, 0.999`** — are on Google Drive:

**➜ [Download the checkpoints](https://drive.google.com/file/d/13Jrpk123862GnNaMt3URGU3mWysRmkwF/view?usp=sharing)**

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--train` / `--test` | — | Mode; one is required |
| `--data` | `data/dataset.csv` | Path to the raw activity CSV |
| `--alpha` | `0.3` | Weight of the coherence term in the loss |
| `--seed` | `42` | Random seed; also names the checkpoint |
| `--batch` | `32` | Batch size |
| `--epochs` | `200` | Maximum epochs |
| `--lr` | `1e-3` | AdamW learning rate |
| `--max-lr` | `3e-4` | OneCycleLR peak |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--patience` | `50` | Early-stopping patience on validation loss |
| `--checkpoint` | seed/α-derived | Checkpoint path for `--test` |

Training writes the best-validation checkpoint to
`checkpoint/model_seed_{seed}_alpha_{alpha}.pth` and reloads it before the test
evaluation.


## Repository layout

```
run.py                        Train / evaluate entry point
common/
├── arguments.py              CLI definition
├── hierarchical_model.py     HierSTT — top-down conditioning across levels
├── st_transformer.py         Spatio-temporal encoder / decoder blocks
├── tft.py                    Temporal Fusion Transformer (national level)
├── loss.py                   Coherence-aware objective
└── metrics.py                WAPE, HAgE, aggregation helpers
data/
├── prepare_data.py           Builds the three hierarchy levels
├── sequences.py              Chronological splits, sliding windows, Datasets
└── DATA.md                   Dataset schema and sources
checkpoint/
└── MODELS.md                 Pretrained checkpoint reference
```

## Requirements

Python 3.9+, PyTorch 2.0+. See `requirements.txt`. A GPU is recommended —
experiments in the paper used an NVIDIA RTX A6000 — but everything runs on CPU.

## Citation
If you use this model in your research, please cite our paper: 
```
@article{lino2026hierarchical,
  title={Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting},
  author={Lino, Filipa and Tavares, B{\'a}rbara and Santiago, Carlos and Soares, Cl{\'a}udia and Marques, Manuel},
  journal={arXiv preprint arXiv:2607.27106},
  year={2026}
}
```

## Acknowledgements
This work was supported by Fundação para a Ciência e a Tecnologia (FCT)
through LARSyS funding (DOIs: 10.54499/LA/P/0083/2020, 10.54499/UIDP/50009/2020, and 10.54499/UIDB/50009/2020),
and PhD grant 2025.03757.BD (DOI: 10.54499/2025.03757.BD).
