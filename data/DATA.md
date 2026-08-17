# Data

**The dataset used in this work is not currently published.**

It was assembled from public Portuguese sources, but redistribution of the
combined table has not yet been cleared. This document describes the schema the code expects.

If redistribution is cleared later, a download link will be added here and to the
[README](../README.md).

The **pretrained weights are released** — see [`MODELS.md`](../checkpoint/MODELS.md) — so the
published models can be inspected and loaded without the data. Running
`--test`, however, needs a schema-compatible CSV to build the test windows from.

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

Derived automatically in `data/prepare_data.py`:

- `month`, `weekday` —  recomputed from `time`;
- `open` — `1` if `M1 != 0`, else `0`. Some EDs genuinely close on certain days;
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

If your data has a different number of hospitals or regions, update
`region_to_hospitals` in both files — the counts must sum to the number of
hospital series, and hospitals must be ordered by region.

## Preprocessing and splits

- Rows before **2021-08-01** are dropped, excluding the atypical COVID-19 demand
  regime. Raw coverage runs 2021-01-01 to 2024-04-20.
- Missing values at specific timestamps are filled by linear interpolation
  **before** the CSV reaches this code.
- Windows: **42-day encoder**, **28-day horizon**, sliding with stride 1. For a
  split of *N* consecutive days this yields `N − 42 − 28 + 1` samples.
- Splits are strictly chronological, no shuffling:

| Split | Range | Days | Samples |
|---|---|---|---|
| Train | 2021-08-01 → 2023-07-15 | 713 | 644 |
| Validation | 2023-07-16 → 2023-12-02 | 140 | 71 |
| Test | 2023-12-03 → 2024-04-20 | 140 | 71 |

> **Implementation note.** `prepare_data.py` converts `time` to integer category
> codes (a 0-based day index), and `create_data_splits` compares those codes
> against day offsets from the start date (713 and 853). The two are coupled: if
> you change how `time` is encoded, update the split thresholds to match.

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

