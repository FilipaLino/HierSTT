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

# Train HierSTT end to end
python scripts/run.py --train --data data/dataset.csv 

# evaluate a saved checkpoint, exporting predictions for further analysis
python scripts/run.py --test --model tft_st --data data/dataset.csv --save-mat
```

That runs the complete pipeline — level construction, windowing, training,
coherence metrics. Swap in your own CSV (schema in [`data/DATA.md`](data/DATA.md)) to get meaningful numbers.

Useful flags: `--alpha` (coherence weight, default `0.3`), `--epochs`,
`--batch`, `--patience`, `--seed`, `--checkpoint-dir`, `--history` (writes the
training curve as JSON). Run `--help` for the rest.

### Reproduce the seed × α grid

The coherence-weight ablation and the mean ± std results come from training
across 6 seeds and 6 values of α:

```bash
python scripts/sweep.py --data data/dataset.csv
```

One row of test metrics per run is appended to
`results/seed_alpha_results.csv`, so an interrupted sweep resumes with
`--skip-existing`. A summary of mean ± std per α is printed at the end.

```bash
# a single cell of the grid
python scripts/sweep.py --data data/dataset.csv --seeds 42 --alphas 0.3
```

## Repository layout

```
hierstt/
├── hierarchy.py          Hierarchy shape (81 hospitals / 5 regions / 1 national)
├── pipeline.py           Raw CSV → levels → windows → scaled tensors → loaders
├── engine.py             Training and evaluation loops
├── losses.py             Coherence-aware objective
├── metrics.py            WAPE, HAgE, aggregation helpers
├── data/
│   ├── prepare_data.py   Builds the three hierarchy levels
│   ├── sequences.py      Chronological splits, sliding windows, Datasets
│   └── scaling.py        All scaler fitting, isolated for auditability
└── models/
    ├── tft.py            Temporal Fusion Transformer (national level)
    ├── transformer.py    Transformer blocks
    ├── spatio_temporal.py  HierSTT  ← the paper model
    └── baselines.py      Hierarchical LSTM / Transformer / CNN ablations

scripts/
├── run.py                Train / evaluate one configuration
├── sweep.py              Seed × α grid
├── make_synthetic_data.py
└── check_no_data.py      Commit guard
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
