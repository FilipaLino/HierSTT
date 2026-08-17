"""Evaluation metrics, all computed in count space after inverting every transform.

* **WAPE** -- scale-normalised accuracy, comparable across hierarchy levels.
  Preferred over MAPE because closed EDs produce zero-demand days for which
  MAPE is undefined.
* **HAgE** -- Hierarchical Aggregation Error: the WAPE between aggregated
  lower-level forecasts and the level above. Reported both against the direct
  higher-level *prediction* (pure coherence; 0 means perfectly coherent) and
  against the observed *ground truth* (does the coherence land on the truth).
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Mapping of region codes to the number of hospitals in each, as per the new hospital order
region_to_hospitals = {0: 10, 1: 5, 2: 18, 3: 21, 4: 27}
num_regions = 5
num_hospitals = 81

def weighted_percentage_error(actual, predicted):
    """WAPE error computed per sample over the complete output vector."""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.sum(np.abs((actual - predicted)),axis=1) / np.sum(np.abs(actual), axis=1)
    return res

def wape(actual, predicted):
    """Mean WAPE over a batch, as a fraction (multiply by 100 for percent)."""
    return float(np.mean(np.abs(weighted_percentage_error(np.asarray(actual), np.asarray(predicted)))))
  
  
def level_metrics(gt, pred):
    return (mean_absolute_error(gt, pred),
            float(np.sqrt(mean_squared_error(gt, pred))),
            wape(gt, pred))


def aggregate_hospitals_to_regions(h_pred_3d):
    """Sum hospital forecasts within each region."""
    batch, _, horizon = h_pred_3d.shape
    out = np.zeros((batch, num_regions, horizon), dtype=h_pred_3d.dtype)
    idx = 0
    for region, num_hosp in region_to_hospitals.items():
        out[:, region, :] = np.sum(h_pred_3d[:, idx:idx + num_hosp, :], axis=1)
        idx += num_hosp
    return out
  
def hierarchical_aggregation_errors(pred_n, pred_r_3d, pred_h_3d, gt_n, gt_r_3d):
    """HAgE for the three aggregation transitions."""
    agg_reg = aggregate_hospitals_to_regions(pred_h_3d)
    agg_nat_from_reg = np.sum(pred_r_3d, axis=1)
    agg_nat_from_hosp = np.sum(agg_reg, axis=1)

    out = {
        "hosp_to_reg": wape(pred_r_3d, agg_reg),
        "hosp_to_nat": wape(pred_n, agg_nat_from_hosp),
        "reg_to_nat": wape(pred_n, agg_nat_from_reg),
        "hosp_to_reg_true": wape(gt_r_3d, agg_reg),
        "hosp_to_nat_true": wape(gt_n, agg_nat_from_hosp),
        "reg_to_nat_true": wape(gt_n, agg_nat_from_reg),
    }
   
    return out
