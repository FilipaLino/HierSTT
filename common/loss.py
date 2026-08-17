"""Hierarchical coherence-aware training objective.

    L = (1 - alpha) * (L_nat + L_reg + L_hosp) + alpha * L_coh
"""

import torch
import torch.nn.functional as F

# Mapping of region codes to the number of hospitals in each, as per the new hospital order
region_to_hospitals = {0: 10, 1: 5, 2: 18, 3: 21, 4: 27}
num_regions = 5
num_hospitals = 81

def inverse_scale_torch(x, scaler):
    center = torch.as_tensor(scaler.center_, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
    return x * scale + center

def hierarchical_loss(n_true_scaled, r_true_scaled, h_true_scaled, n_pred_scaled, r_pred_scaled, h_pred_scaled, alpha, scaler_y_n, scaler_y_r, scaler_y_h): 
    # Direct losses in scaled space
    loss_pt = F.smooth_l1_loss(n_pred_scaled, n_true_scaled)
    loss_r = F.smooth_l1_loss(r_pred_scaled, r_true_scaled)
    loss_h = F.smooth_l1_loss(h_pred_scaled, h_true_scaled)
    
    n_pred_log = inverse_scale_torch(n_pred_scaled, scaler_y_n)
    r_pred_log  = inverse_scale_torch(r_pred_scaled, scaler_y_r)
    h_pred_log  = inverse_scale_torch(h_pred_scaled, scaler_y_h)
    
    # Ensure sum constraints
    batch = h_pred_log.shape[0]
    out_seq = n_pred_log.shape[-1]
    r_pred_log = r_pred_log.view(batch, num_regions, out_seq)
    h_pred_log = h_pred_log.view(batch, num_hospitals, out_seq)

    # Convert to count space for coherence
    r_pred  = torch.expm1(r_pred_log).clamp_min(0.0)
    h_pred  = torch.expm1(h_pred_log).clamp_min(0.0)
    
    # Regional consistency: each region vs the sum of its own hospitals
    idx, loss_sum_r = 0, 0            
    for region, num_hosp in region_to_hospitals.items(): 
        sub_r_pred_log = r_pred_log[:, region, :]
        sub_h_pred = h_pred[:, idx:idx + num_hosp, :]
        idx += num_hosp
        loss_sum_r += F.smooth_l1_loss(sub_r_pred_log, torch.log1p(torch.sum(sub_h_pred,dim=1)))
     
    constraint_loss = F.smooth_l1_loss(n_pred_log, torch.log1p(torch.sum(r_pred,dim=1))) + F.smooth_l1_loss(n_pred_log, torch.log1p(torch.sum(h_pred, dim=1))) + (loss_sum_r/r_pred.shape[1])
        
    return ((1-alpha)*(loss_pt + loss_r + loss_h)) + (alpha*constraint_loss)
