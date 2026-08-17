import torch
import torch.nn as nn
import math

class SpatioTemporalBlock(nn.Module):
    """
    One layer of alternating temporal self-attention + spatial self-attention.
    Input/output shape: [B, E, T, d_model]
    """
    def __init__(self, d_model, nhead, dropout=0.1, ff_mult=4):
        super().__init__()
        dim_ff = d_model * ff_mult

        # --- Temporal attention (each entity attends over its own time axis) ---
        self.temp_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temp_norm  = nn.LayerNorm(d_model)
        self.temp_drop  = nn.Dropout(dropout)

        # --- Spatial attention (all entities attend to each other per timestep) ---
        self.spat_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.spat_norm  = nn.LayerNorm(d_model)
        self.spat_drop  = nn.Dropout(dropout)

        # --- Shared feed-forward ---
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x, temp_mask=None):
        """
        x: [B, E, T, d_model]
        Returns: [B, E, T, d_model]
        """
        B, E, T, D = x.shape

        # ---- Temporal attention: merge B and E, attend over T ----
        x_t = x.reshape(B * E, T, D)                    # [B*E, T, D]
        attn_out, _ = self.temp_attn(x_t, x_t, x_t, attn_mask=temp_mask)
        x_t = self.temp_norm(x_t + self.temp_drop(attn_out))
        x = x_t.reshape(B, E, T, D)

        # ---- Spatial attention: merge B and T, attend over E ----
        x_s = x.permute(0, 2, 1, 3).reshape(B * T, E, D)   # [B*T, E, D]
        attn_out, _ = self.spat_attn(x_s, x_s, x_s)
        x_s = self.spat_norm(x_s + self.spat_drop(attn_out))
        x = x_s.reshape(B, T, E, D).permute(0, 2, 1, 3)    # [B, E, T, D]

        # ---- Feed-forward (shared across E and T) ----
        x_ff = x.reshape(B * E * T, D)
        x_ff = self.ff_norm(x.reshape(B * E * T, D) + self.ff(x_ff))
        x = x_ff.reshape(B, E, T, D)

        return x


class SpatioTemporalEncoder(nn.Module):
    """
    Projects input features → d_model, then applies N spatio-temporal blocks.
    Returns the full encoded sequence: [B, E, T, d_model]
    """
    def __init__(self, input_size, d_model, nhead, num_layers, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_size, d_model)

        # Temporal positional encoding
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0).unsqueeze(0))  # [1, 1, T, D]

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [B, E, T, F]
        Returns: [B, E, T, d_model]
        """
        B, E, T, F = x.shape
        x = self.input_proj(x)                   # [B, E, T, d_model]
        x = x + self.pe[:, :, :T, :]             # add temporal PE
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SpatioTemporalDecoder(nn.Module):
    """
    Decodes future predictions using:
      - Future known covariates (month, weekday, holiday, open)
      - Higher-level context (e.g. national forecast for regional decoder)
      - Cross-attention to the encoder memory
    Output: [B, E, pred_days]
    """
    def __init__(self, d_model, nhead, num_layers, pred_days,
                 num_entities, max_num_entities,higher_ctx_dim=1, dropout=0.1):
        super().__init__()
        self.d_model    = d_model
        self.pred_days  = pred_days
        self.num_entities = num_entities

        # Categorical embeddings for future-known covariates
        self.month_emb   = nn.Embedding(13, d_model)
        self.weekday_emb = nn.Embedding(7,  d_model)
        self.holiday_emb = nn.Embedding(max_num_entities + 1, d_model)
        self.open_emb    = nn.Embedding(max_num_entities + 1, d_model)
        self.entity_emb = nn.Embedding(num_entities, d_model)

        # Learned step bias
        self.query_bias = nn.Parameter(torch.randn(pred_days, d_model) * 0.02)

        # Higher-level context projection (accepts 1 or 2 signals concatenated)
        self.higher_proj = nn.Linear(higher_ctx_dim, d_model)

        # Temporal PE for decoder queries
        pe = torch.zeros(pred_days, d_model)
        pos = torch.arange(pred_days).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("dec_pe", pe)  # [pred_days, d_model]

        # Cross-attention + spatial attention decoder blocks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn"  : nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                "self_norm"  : nn.LayerNorm(d_model),
                "cross_attn" : nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                "cross_norm" : nn.LayerNorm(d_model),
                "spat_attn"  : nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True),
                "spat_norm"  : nn.LayerNorm(d_model),
                "ff"         : nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
                ),
                "ff_norm"    : nn.LayerNorm(d_model),
            }))

        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, memory, future_cat, higher_context):
        """
        memory:         [B, E, T, d_model]  — from SpatioTemporalEncoder
        future_cat:     [B, E, pred_days, 4]  — (month, weekday, holiday, open)
        higher_context: [B, E, pred_days]   — national or national+regional signal

        Returns: [B, E, pred_days]
        """
        B, E, T, D = memory.shape

        # Build decoder queries from future covariates
        month   = future_cat[..., 0].clamp(0, 12).long()     # [B, E, P]
        weekday = future_cat[..., 1].clamp(0, 6).long()
        holiday = future_cat[..., 2].long()
        isopen  = future_cat[..., 3].long()
        entity_ids = torch.arange(E, device=future_cat.device).view(1, E, 1).expand(B, E, self.pred_days)

        q = (self.month_emb(month) + self.weekday_emb(weekday)
             + self.holiday_emb(holiday) + self.open_emb(isopen) + self.entity_emb(entity_ids))   # [B, E, P, D]
        q = q + self.dec_pe.unsqueeze(0).unsqueeze(0)               # temporal PE
        q = q + self.query_bias.unsqueeze(0).unsqueeze(0)           # learned bias

        # Inject higher-level context
        ctx = self.higher_proj(torch.tanh(higher_context))
        q = q + ctx                                                  # [B, E, P, D]

        # Flatten memory for cross-attention key/value: [B*E, T, D]
        mem_flat = memory.reshape(B * E, T, D)
        q_flat   = q.reshape(B * E, self.pred_days, D)

        for layer in self.layers:
            # Self-attention over prediction horizon (temporal)
            sa, _ = layer["self_attn"](q_flat, q_flat, q_flat)
            q_flat = layer["self_norm"](q_flat + sa)

            # Cross-attention to encoder memory
            ca, _ = layer["cross_attn"](q_flat, mem_flat, mem_flat)
            q_flat = layer["cross_norm"](q_flat + ca)

            # Reshape back, spatial attention over entities
            q_4d = q_flat.reshape(B, E, self.pred_days, D)
            q_sp = q_4d.permute(0, 2, 1, 3).reshape(B * self.pred_days, E, D)
            spa, _ = layer["spat_attn"](q_sp, q_sp, q_sp)
            q_sp = layer["spat_norm"](q_sp + spa)
            q_flat = q_sp.reshape(B, self.pred_days, E, D).permute(0, 2, 1, 3).reshape(B * E, self.pred_days, D)

            # Feed-forward
            ff_in = q_flat.reshape(B * E * self.pred_days, D)
            q_flat = layer["ff_norm"](q_flat + layer["ff"](ff_in).reshape(B * E, self.pred_days, D))

        out = self.out_proj(q_flat).squeeze(-1)    # [B*E, pred_days]
        return out.reshape(B, E, self.pred_days)   # [B, E, pred_days]