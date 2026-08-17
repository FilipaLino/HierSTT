# Pretrained models

All checkpoints from the paper are released on Google Drive.

**➜ [Download folder](https://drive.google.com/file/d/13Jrpk123862GnNaMt3URGU3mWysRmkwF/view?usp=sharing)**

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

## Licence and citation

The weights are released under the same MIT terms as the code. If you use them,
please cite the paper — see the [README](../README.md).
