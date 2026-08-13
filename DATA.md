# Data

**No dataset is distributed with this repository.** The Portuguese ED activity
data used in the paper was assembled from public sources but is not
redistributed here. This document describes the schema the code expects.

---

## Expected input

One CSV, one row per **(hospital, day)**. Pass its path with `--data`.

| Column | Type | Level | Description |
|---|---|---|---|
| `time` | date (`YYYY-MM-DD`) | all | Observation day |
| `D0` | string/int | hospital | Hospital identifier |
| `RHA` | string | hospital | Regional health administration the hospital belongs to |
| `locality` | string | hospital | Municipality or locality |
| `M1` | int | all | **Target.** Total urgent episodes that day |
| `M2`–`M8` | int | all | Manchester triage counts: blue, white, green, no triage, yellow, orange, red |
| `waiting_time` | float | regional, national | Mean minutes between triage and medical evaluation |
| `SUB`, `SUMC`, `SUP`, `SUPCT` | int/bool | hospital | Emergency service type (basic, medical-surgical, polyvalent, trauma centre) |
| `age_range` | categorical | hospital | Admitted age band |
| `patient_access` | int | hospital | Self-admissions |
| `health24_access` | int | hospital | Referrals via the SNS24 phone line |
| `PHC_access` | int | hospital | Referrals from primary health care |
| `emergency_doc_access` | int | hospital | Referrals from emergency doctors |
| `hospital_doc_access` | int | hospital | Referrals from hospital doctors |
| `is_holiday` | 0/1 | all | Public holiday indicator |
| `AQI` | float | regional | Air quality index |
| `tmpc` | float | regional | Temperature (°C) |
| `mortality` | float | national | Daily mortality count |

Derived automatically in `hierstt/data/prepare_data.py`:

- `month`, `weekday` — from `time`
- `open` — `1` if `M1 != 0`, else `0`. Some EDs genuinely close on some days;
  this flag is a **future-known** decoder covariate, which is what lets the model
  predict near-zero demand on closure days.

`AQI`, `tmpc` and `mortality` are only used after aggregation, so hospital-level
values may be constant within a region (or within the country) without harm.

## How the three levels are built

| Level | Series | Construction |
|---|---|---|
| Hospital | 81 | Raw rows, minus `AQI`, `waiting_time`, `tmpc`, `mortality` |
| Regional | 5 | Grouped by `(time, RHA)`: counts summed, `waiting_time`/`AQI`/`tmpc` averaged |
| National | 1 | Grouped by `time`: counts summed, `waiting_time`/`mortality` averaged, holidays from the `holidays` package |

This produces the heterogeneous feature space described in the paper: 26 / 17 /
16 features at the hospital / regional / national levels.

### Hierarchy shape

The published hierarchy is 81 hospitals in 5 RHAs. Region indices follow the
**alphabetical order of the `RHA` labels**, because
`pandas.Categorical.cat.codes` assigns codes that way:

| Index | RHA | Hospitals |
|---|---|---|
| 0 | Alentejo | 10 |
| 1 | Algarve | 5 |
| 2 | Centro | 18 |
| 3 | Lisboa e Vale do Tejo | 21 |
| 4 | Norte | 27 |

`hierstt/hierarchy.py` infers this from your data at load time
(`Hierarchy.from_dataframe`), so a different number of hospitals or regions works
without code changes.

## Preprocessing and splits

- Rows before `--start-date` (default `2021-08-01`) are dropped, to exclude the
  atypical COVID-19 demand regime.
- Missing values at specific timestamps are filled by linear interpolation
  **before** the CSV reaches this code.
- Splits are strictly chronological with no shuffling: train up to
  `--train-end` (default `2023-07-15`), validation up to `--val-end` (default
  `2023-12-02`), test thereafter. On the published data that is 713 / 140 / 140
  days, yielding 644 / 71 / 71 sliding-window samples.
- Targets are `log1p`-transformed; features and targets are scaled with
  `RobustScaler` **fitted on the training split only**
  (`hierstt/data/scaling.py`). Metrics are computed in count space after
  inverting every transform.

> **Implementation note.** `prepare_data.py` converts `time` to integer category
> codes (a 0-based day index), and `create_data_splits` compares those codes
> against day offsets from `--start-date`. The two are coupled: if you change how
> `time` is encoded, update the split thresholds to match.

## Original sources

All public, all free to access. Combining and cleaning them is the work; the
result is what is not redistributed here.

| Source | Used for |
|---|---|
| [Transparência SNS](https://transparencia.sns.gov.pt/explore/) | ED activity, triage, access pathways |
| [SNS ED monitoring](https://www.sns.gov.pt/monitorizacao-do-sns/servicos-de-urgencia/) | Daily ED demand and waiting times |
| [AQICN Portugal](https://aqicn.org/map/portugal/pt/) | Air quality index |
| [Iowa State Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml?network=PT__ASOS) | Temperature |
| [EVM, Ministério da Saúde](https://evm.min-saude.pt/) | Mortality |

The AQI is computed following the US EPA technical assistance document for
reporting the daily Air Quality Index.

## If you are adding your own data

1. Put the CSV anywhere; `data/` is gitignored so files there cannot be
   committed by accident.
2. Run `python scripts/check_no_data.py --install-hook` once. It blocks commits
   containing data-like files, including via `git add -f`.
3. Sanity-check the schema by running the pipeline for one epoch:
   `python scripts/run.py --train --data your.csv --epochs 1`.
