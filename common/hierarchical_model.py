import torch
import torch.nn as nn
import math

from .tft import TFT 
from .st_transformer import SpatioTemporalEncoder, SpatioTemporalDecoder


class HierSTT(nn.Module):
    """
    Drop-in replacement for HierarchicalTrans that uses spatio-temporal
    transformers at regional and hospital levels.

    Signature compatible with the tft_trans branch in your training loop.
    """
    def __init__(self, config, r_input_size, h_input_size,
                 d_model, nhead, num_layers, output_size, dropout_rate=0.3):
        super().__init__()
        self.pred_days     = 28
        self.num_regions   = 5
        self.num_hospitals = 81

        # Map each hospital to its region index
        hospital_to_region = []
        for reg, n_hosp in {0: 10, 1: 5, 2: 18, 3: 21, 4: 27}.items():
            hospital_to_region.extend([reg] * n_hosp)
        self.register_buffer("hospital_to_region",
                             torch.tensor(hospital_to_region, dtype=torch.long))

        feat_per_region   = r_input_size // self.num_regions
        feat_per_hospital = h_input_size // self.num_hospitals

        # National level (keep your TFT unchanged)
        self.pt_tft = TFT(config)

        # Regional encoder/decoder
        self.r_encoder = SpatioTemporalEncoder(feat_per_region, d_model, nhead, num_layers,
                                               dropout=dropout_rate)
        self.r_decoder = SpatioTemporalDecoder(d_model, nhead, num_layers, self.pred_days,
                                               num_entities=self.num_regions,
                                               higher_ctx_dim=1,
                                               max_num_entities=27,
                                               dropout=dropout_rate)

        # Hospital encoder/decoder
        self.h_encoder = SpatioTemporalEncoder(feat_per_hospital, d_model, nhead, num_layers,
                                               dropout=dropout_rate)
        self.h_decoder = SpatioTemporalDecoder(d_model, nhead, num_layers, self.pred_days,
                                               num_entities=self.num_hospitals,
                                               higher_ctx_dim=1,
                                               max_num_entities=27,
                                               dropout=dropout_rate)

        self.feat_per_region   = feat_per_region
        self.feat_per_hospital = feat_per_hospital

    def _build_future_cat(self, x_fut, num_entities, pred_days):
        """
        x_fut: [B, num_entities * pred_days, 4]  (from your existing data pipeline)
        Returns: [B, num_entities, pred_days, 4]
        """
        B = x_fut.size(0)
        return x_fut.reshape(B, num_entities, pred_days, x_fut.size(-1)).long()

    def forward(self, x_pt, x_r, x_r_fut, x_h, x_h_fut):
        B   = x_r.size(0)
        T_r = x_r.size(1)
        T_h = x_h.size(1)

        # ---- National ----
        pt_out  = self.pt_tft(x_pt)          # [B, 28, 1]
        pt_pred = pt_out.squeeze(-1)          # [B, 28]

        # ---- Regional ----
        # Reshape history: [B, T, 5*F] → [B, 5, T, F]
        x_r_4d = x_r.view(B, T_r, self.num_regions, self.feat_per_region)\
                     .permute(0, 2, 1, 3)                       # [B, 5, T, F]
        r_memory = self.r_encoder(x_r_4d)                       # [B, 5, T, d_model]

        # Future covariates for regions
        r_fut_4d = self._build_future_cat(x_r_fut, self.num_regions, self.pred_days)

        # Higher context: broadcast national forecast to all regions
        # [B, 28] → [B, 5, 28]
        pt_ctx_r = pt_pred.unsqueeze(1).expand(B, self.num_regions, self.pred_days).unsqueeze(-1) # [B, 5, 28, 1]

        r_pred_3d = self.r_decoder(r_memory, r_fut_4d, pt_ctx_r)  # [B, 5, 28]
        r_pred_flat = r_pred_3d.reshape(B, self.num_regions * self.pred_days)

        # ---- Hospital ----
        # Reshape history: [B, T, 81*F] → [B, 81, T, F]
        x_h_4d = x_h.view(B, T_h, self.num_hospitals, self.feat_per_hospital)\
                     .permute(0, 2, 1, 3)                        # [B, 81, T, F]
        h_memory = self.h_encoder(x_h_4d)                        # [B, 81, T, d_model]

        # Future covariates for hospitals
        h_fut_4d = self._build_future_cat(x_h_fut, self.num_hospitals, self.pred_days)

        # Higher context: national + corresponding region forecast
        # pt: [B, 28] → [B, 81, 28]
        pt_ctx_h = pt_pred.unsqueeze(1).expand(B, self.num_hospitals, self.pred_days)
        # region: [B, 5, 28] → [B, 81, 28] via hospital_to_region mapping
        r_ctx_h  = r_pred_3d[:, self.hospital_to_region, :]      # [B, 81, 28]

        # Fuse
        hosp_ctx = r_ctx_h.unsqueeze(-1)
        
        h_pred_3d = self.h_decoder(h_memory, h_fut_4d, hosp_ctx) # [B, 81, 28]
        h_pred_flat = h_pred_3d.reshape(B, self.num_hospitals * self.pred_days)

        return pt_pred, r_pred_flat, h_pred_flat
