"""Train/evaluate the HierSTT model.

The original study dataset is intentionally external to the repository. This
script expects a local CSV path and never writes raw/processed data.

Key safeguards
--------------
1. Temporal splits use real datetimes.
2. The forecast horizon never contains M1 or any other unknown future outcome.
3. Only deterministic future covariates are exposed to the decoders:
   month, weekday, holiday status, and national ``time_idx``.
4. All scalers are fitted on training data only.
5. Test evaluation requires a separate ``--test`` invocation.
"""
import os, random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset, DataLoader

from data.prepare_data import process_data
from data.sequences import HierarchicalMixedDataset, TFTDataset, create_data_splits, create_sequences
from common.arguments import parse_args
from common.hierarchical_model import HierSTT
from common.loss import hierarchical_loss
from common.metrics import level_metrics, hierarchical_aggregation_errors

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Check for CUDA and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------
SEED = 42
START_DATE = pd.Timestamp("2021-08-01")
TRAIN_END = pd.Timestamp("2023-07-15")
VAL_END = pd.Timestamp("2023-12-02")

ENCODER_LENGTH = 6 * 7
PREDICTION_LENGTH = 4 * 7

STATIC_COLS = ["dummy_zero"]
# Known at prediction time, national level
CATEGORICAL_COLS = ["month", "weekday", "is_holiday"]
# Known at prediction time, regional and hospital levels (adds the open/closed flag)
HR_CATEGORICAL_COLS = ["month", "weekday", "is_holiday", "open"]

# Observed in the past. 'time' and 'open' are also known in the future.
REAL_COLS = ["waiting_time", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8",
             "mortality", "time", "open"]

# Historical real variables available to the national encoder. ``time_idx`` is
# placed last because it is the ONLY real variable made available to the TFT
# decoder over the forecast horizon.
NATIONAL_REAL_COLS = [
    "waiting_time",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "mortality",
    "time_idx",
]
NATIONAL_FUTURE_REAL_COLS = ["time_idx"]

# Mapping of region codes to the number of hospitals in each, as per the new hospital order
region_to_hospitals = {0: 10, 1: 5, 2: 18, 3: 21, 4: 27}


# Reproducibility 
def set_seed(seed=47):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    set_seed(42)
    
    # Parse arguments
    args = parse_args()
    print(args)
    
    # Preparing data
    data = pd.read_csv(args.data)
    # Sorting hospital_level by 'RHA' alphabetically
    if 'RHA' in data.columns:
        data = data.sort_values(by=['RHA', 'time'],ignore_index=True)

    # Filter out everything before 2021-08-01 (covid)
    data['time'] = pd.to_datetime(data['time'], errors='coerce')
    cutoff = pd.Timestamp('2021-08-01')
    data = data[data['time'] >= cutoff].reset_index(drop=True).copy()
             
    # Process the data using the imported function
    hospital_level, rha_level, national_level = process_data(data)
    national_level = national_level.copy()
    national_level['dummy_zero'] = 0 # categorical variable for national level
   
    # Mappings between array positions and actual IDs 
    region_ids = sorted(rha_level['RHA'].unique().tolist())
    region_idx_to_id = {idx: rha_id for idx, rha_id in enumerate(region_ids)}

    # Keep the same hospital ordering used in create_sequences(level='Hospital')
    hospital_ids = hospital_level['D0'].unique().tolist()
    hospital_idx_to_id = {idx: hosp_id for idx, hosp_id in enumerate(hospital_ids)}
    # Optional: hospital index -> region ID
    hospital_idx_to_region_id = {}
    for idx, hosp_id in hospital_idx_to_id.items():
        hosp_region = hospital_level.loc[hospital_level['D0'] == hosp_id, 'RHA'].iloc[0]
        hospital_idx_to_region_id[idx] = hosp_region
    
    # ------------------------ Chronological splits (thresholds are day offsets from start_date) ----------------------------
    train_threshold = (TRAIN_END - START_DATE).days  # 713
    val_threshold   = (VAL_END- START_DATE).days  # 853
    
    train_h, val_h, test_h = create_data_splits(hospital_level, train_threshold, val_threshold)
    train_r, val_r, test_r = create_data_splits(rha_level, train_threshold, val_threshold)
    train_n, val_n, test_n = create_data_splits(national_level, train_threshold, val_threshold)
    
    def seq_nat(df):
        return create_sequences(df, ENCODER_LENGTH, PREDICTION_LENGTH, "National", STATIC_COLS, CATEGORICAL_COLS, REAL_COLS, log_target=True)

    def seq_lower(df, level):
        return create_sequences(df, ENCODER_LENGTH, PREDICTION_LENGTH, level, STATIC_COLS, HR_CATEGORICAL_COLS, REAL_COLS, log_target=True)
    
    X_train_n, y_train_n = seq_nat(train_n)
    X_val_n,   y_val_n = seq_nat(val_n)
    X_test_n,  y_test_n = seq_nat(test_n)

    X_train_r, X_train_fut_r, y_train_r = seq_lower(train_r, "Regional")
    X_val_r,   X_val_fut_r,   y_val_r   = seq_lower(val_r, "Regional")
    X_test_r,  X_test_fut_r,  y_test_r  = seq_lower(test_r, "Regional")

    X_train_h, X_train_fut_h, y_train_h = seq_lower(train_h, "Hospital")
    X_val_h,   X_val_fut_h,   y_val_h   = seq_lower(val_h, "Hospital")
    X_test_h,  X_test_fut_h,  y_test_h  = seq_lower(test_h, "Hospital")
    
    # ------------------------ Scaling: fitted on train only -------------------------
    # National level
    num_ids = len(STATIC_COLS)
    R_enc = len(REAL_COLS)
    
    real_train_n = X_train_n[:, :, num_ids:num_ids+R_enc].reshape(-1, R_enc)
    scaler_real = RobustScaler().fit(real_train_n)

    def scale_real_only(X, scaler):
        X = X.copy()
        real = X[:, :, num_ids:num_ids+R_enc].reshape(-1, R_enc)
        real = scaler.transform(real).reshape(X.shape[0], X.shape[1], R_enc)
        X[:, :, num_ids:num_ids+R_enc] = real
        return X
    
    X_train_n = scale_real_only(X_train_n, scaler_real)
    X_val_n   = scale_real_only(X_val_n,   scaler_real)
    X_test_n  = scale_real_only(X_test_n,  scaler_real)
    
    scaler_y_n = RobustScaler().fit(y_train_n)
        
    # Transform train/val/test consistently 
    def apply_scale(X, scaler):
        F_aux = X.shape[-1]
        return scaler.transform(X.reshape(-1, F_aux)).reshape(X.shape)
    
    y_train_n = apply_scale(y_train_n, scaler_y_n)
    y_val_n   = apply_scale(y_val_n,   scaler_y_n)
    y_test_n  = apply_scale(y_test_n,  scaler_y_n)
    
    
    # Regional and Hospital Levels
    F_r  = X_train_r.shape[-1]   # entities * features_per_entity
    F_h  = X_train_h.shape[-1]
    
    scaler_X_r  = RobustScaler().fit(X_train_r.reshape(-1, F_r))
    scaler_y_r  = RobustScaler().fit(y_train_r)
    scaler_X_h  = RobustScaler().fit(X_train_h.reshape(-1, F_h))
    scaler_y_h  = RobustScaler().fit(y_train_h)
    
    X_train_r = apply_scale(X_train_r, scaler_X_r)
    X_val_r   = apply_scale(X_val_r,   scaler_X_r)
    X_test_r  = apply_scale(X_test_r,  scaler_X_r)
    y_train_r = apply_scale(y_train_r, scaler_y_r)
    y_val_r   = apply_scale(y_val_r,   scaler_y_r)
    y_test_r  = apply_scale(y_test_r,  scaler_y_r)
    
    X_train_h = apply_scale(X_train_h, scaler_X_h)
    X_val_h   = apply_scale(X_val_h,   scaler_X_h)
    X_test_h  = apply_scale(X_test_h,  scaler_X_h)
    y_train_h = apply_scale(y_train_h, scaler_y_h)
    y_val_h   = apply_scale(y_val_h,   scaler_y_h)
    y_test_h  = apply_scale(y_test_h,  scaler_y_h)
    
    # --- Convert to tensors and move to device ---
    def f32(a):
        return torch.tensor(a, dtype=torch.float32)

    def i64(a):
        return torch.tensor(a, dtype=torch.long)
               
    def make_dataset(Xn, yn, Xr, Xrf, yr, Xh, Xhf, yh):
        return HierarchicalMixedDataset(
            TFTDataset(f32(Xn), f32(yn), num_ids),
            TensorDataset(f32(Xr), i64(Xrf), f32(yr)),
            TensorDataset(f32(Xh), i64(Xhf), f32(yh)),
        )

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)
    
    train_dataset = make_dataset(X_train_n, y_train_n, X_train_r, X_train_fut_r,
                                 y_train_r, X_train_h, X_train_fut_h, y_train_h)
    val_dataset   = make_dataset(X_val_n, y_val_n, X_val_r, X_val_fut_r, y_val_r,
                                 X_val_h, X_val_fut_h, y_val_h)
    test_dataset  = make_dataset(X_test_n, y_test_n, X_test_r, X_test_fut_r, y_test_r,
                                 X_test_h, X_test_fut_h, y_test_h)
    
    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader   = make_loader(val_dataset, shuffle=False)
    test_loader  = make_loader(test_dataset, shuffle=False)
              
    # --- TFT config for the national level ---
    config = {
        "static_variables": len(STATIC_COLS),
        "static_embedding_vocab_sizes": [int(train_n[c].max()) + 1 for c in STATIC_COLS],
        "time_varying_categoical_variables": len(CATEGORICAL_COLS),
        "time_varying_real_variables_encoder": len(REAL_COLS),
        "time_varying_real_variables_decoder": 2, # the decoder only sees the future of the 'time' and 'open' real variables
        "num_masked_series": len(REAL_COLS) - 2,
        "time_varying_embedding_vocab_sizes": [int(train_n[c].max()) + 1 for c in CATEGORICAL_COLS],
        "embedding_dim": 8,
        "lstm_hidden_dimension": 128,
        "lstm_layers": 2,
        "dropout": 0.1,
        "device": device,
        "batch_size": args.batch,
        "encode_length": ENCODER_LENGTH,
        "seq_length": ENCODER_LENGTH + PREDICTION_LENGTH,
        "attn_heads": 4,
        "num_quantiles": 1,
        "vailid_quantiles": [0.5],
    }
    
    # Model initialization
    input_sizes = [X_train_r.shape[-1], X_train_h.shape[-1]]
    outputs = [y_train_r.shape[-1], y_train_h.shape[-1]]
    steps_per_epoch = len(train_loader)
    hp = dict(d_model=128, nhead=4, num_layers=2, dropout=0.2)
    model = HierSTT(config, input_sizes[0], input_sizes[1], hp["d_model"], hp["nhead"], hp["num_layers"], outputs, hp["dropout"]).to(device)
       
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=args.max_lr, total_steps=args.epochs * steps_per_epoch, 
                                                    pct_start=0.05, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4) 
        
    best_val_loss = float('inf')
    best_model_path = f"checkpoint/model_seed_{args.seed}_alpha_{str(args.alpha).replace('.', '_')}.pth"  # Path where the best model will be saved
    best_epoch = -1
    best_model = None
    
    if args.train:        
        for epoch in range(args.epochs):
            model.train()
            total_loss, total_samples = 0.0, 0
            for n_data, n_labels, r_data, r_fut, r_labels, h_data, h_fut, h_labels in train_loader:
                n_data = {k: v.to(device) for k, v in n_data.items()}
                n_labels = n_labels.to(device)
                r_data, r_fut, r_labels = r_data.to(device), r_fut.to(device), r_labels.to(device)
                h_data, h_fut, h_labels = h_data.to(device), h_fut.to(device), h_labels.to(device)
                
                n_pred, r_pred, h_pred = model(n_data, r_data, r_fut, h_data, h_fut)
                                       
                loss = hierarchical_loss(n_labels, r_labels, h_labels, n_pred, r_pred, h_pred, args.alpha,  scaler_y_n, scaler_y_r, scaler_y_h)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() 
                
                total_loss += loss.item() * n_labels.size(0)
                total_samples += n_labels.size(0)
            
            print(f'Epoch {epoch+1}, Loss: {total_loss/total_samples:.2f}')
           
            # Evaluate the model on the val set
            model.eval()
            val = {k: 0.0 for k in
                          ("loss", "mae_n", "mae_r", "mae_h", "rmse_n",
                           "rmse_r", "rmse_h", "wape_n", "wape_r", "wape_h")}
            total_samples = 0
            
            with torch.no_grad():
                for n_data, n_labels, r_data, r_fut, r_labels, h_data, h_fut, h_labels  in val_loader:
                    n_data = {k: v.to(device) for k, v in n_data.items()}
                    n_labels = n_labels.to(device)
                    r_data, r_fut, r_labels = r_data.to(device), r_fut.to(device), r_labels.to(device)
                    h_data, h_fut, h_labels = h_data.to(device), h_fut.to(device), h_labels.to(device)
                    
                    n_pred, r_pred, h_pred = model(n_data, r_data, r_fut, h_data, h_fut)
                    
                    bs = n_labels.size(0)                    
                    val["loss"] += hierarchical_loss(
                        n_labels, r_labels, h_labels, n_pred, r_pred, h_pred, args.alpha,
                        scaler_y_n, scaler_y_r, scaler_y_h).item() * bs  # Multiply by batch size for weighted average
                                        
                    # Metric calculations 
                    gt_n   = np.maximum(np.expm1(scaler_y_n.inverse_transform(n_labels.cpu().numpy().squeeze())), 0.0)  
                    pred_n = np.maximum(np.expm1(scaler_y_n.inverse_transform(n_pred.cpu().numpy().squeeze())), 0.0)  
                    gt_r   = np.maximum(np.expm1(scaler_y_r.inverse_transform(r_labels.cpu().numpy().squeeze())), 0.0)  
                    pred_r = np.maximum(np.expm1(scaler_y_r.inverse_transform(r_pred.cpu().numpy().squeeze())), 0.0)  
                    gt_h   = np.maximum(np.expm1(scaler_y_h.inverse_transform(h_labels.cpu().numpy().squeeze())), 0.0)      
                    pred_h = np.maximum(np.expm1(scaler_y_h.inverse_transform(h_pred.cpu().numpy().squeeze())), 0.0)  
                    
                    for tag, (g, p) in (("n", (gt_n, pred_n)), ("r", (gt_r, pred_r)), ("h", (gt_h, pred_h))):
                        mae, rmse, w = level_metrics(g, p)
                        val[f"mae_{tag}"] += mae * bs
                        val[f"rmse_{tag}"] += rmse * bs
                        val[f"wape_{tag}"] += w * bs              
                    
                    total_samples += bs
                    
            # Averaging the metrics
            val_results = {k: v / max(total_samples, 1) for k, v in val.items()}
            if val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                best_epoch = epoch
                best_model = model.state_dict()
                torch.save(best_model, best_model_path)  # Save model parameters
                print(f"Epoch {epoch + 1:3d}/{args.epochs}  train {total_loss/total_samples:.4f}  "
                  f"val {val_results['loss']:.4f}  WAPE nat {val_results['wape_n'] * 100:.2f}%  "
                  f"reg {val_results['wape_r'] * 100:.2f}%  hosp {val_results['wape_h'] * 100:.2f}%")
            if epoch - best_epoch > args.patience:
                break
        
        model.load_state_dict(torch.load(best_model_path))
        
    elif args.test:
        # Load best model
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Best model checkpoint not found at: {args.checkpoint}. "
                f"Run with --train first or provide the correct checkpoint."
            )

        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    
    # Evaluate the model on the test set
    model.eval()
    test = {k: 0.0 for k in
              ("loss", "mae_n", "mae_r", "mae_h", "rmse_n", "rmse_r", "rmse_h",
               "wape_n", "wape_r", "wape_h",
               "hage_hosp_to_reg", "hage_hosp_to_nat", "hage_reg_to_nat",
               "hage_hosp_to_reg_true", "hage_hosp_to_nat_true", "hage_reg_to_nat_true")}
    total_samples = 0
    num_regions = 5
    num_hospitals = 81
    pred_days = 28
    
    with torch.no_grad():
        for n_data, n_labels, r_data, r_fut, r_labels, h_data, h_fut, h_labels in test_loader:
            n_data = {k: v.to(device) for k, v in n_data.items()}
            n_labels = n_labels.to(device)
            r_data, r_fut, r_labels = r_data.to(device), r_fut.to(device), r_labels.to(device)
            h_data, h_fut, h_labels = h_data.to(device), h_fut.to(device), h_labels.to(device)

            bs = n_labels.size(0)
                           
            n_pred, r_pred, h_pred = model(n_data, r_data, r_fut, h_data, h_fut)
            
            test["loss"] += hierarchical_loss(n_labels, r_labels, h_labels, n_pred, r_pred, h_pred, args.alpha,
                                                          scaler_y_n, scaler_y_r, scaler_y_h).item() * bs
            
            # Metric calculations        
            gt_n   = np.maximum(np.expm1(scaler_y_n.inverse_transform(n_labels.cpu().numpy().squeeze())), 0.0)     
            pred_n = np.maximum(np.expm1(scaler_y_n.inverse_transform(n_pred.cpu().numpy().squeeze())), 0.0)  
            gt_r   = np.maximum(np.expm1(scaler_y_r.inverse_transform(r_labels.cpu().numpy().squeeze())), 0.0)      
            pred_r = np.maximum(np.expm1(scaler_y_r.inverse_transform(r_pred.cpu().numpy().squeeze())), 0.0)  
            gt_h   = np.maximum(np.expm1(scaler_y_h.inverse_transform(h_labels.cpu().numpy().squeeze())), 0.0)             
            pred_h = np.maximum(np.expm1(scaler_y_h.inverse_transform(h_pred.cpu().numpy().squeeze())), 0.0)                 
            
            for tag, (g, p) in (("n", (gt_n, pred_n)), ("r", (gt_r, pred_r)), ("h", (gt_h, pred_h))):
                mae, rmse, w = level_metrics(g, p)
                test[f"mae_{tag}"] += mae * bs
                test[f"rmse_{tag}"] += rmse * bs
                test[f"wape_{tag}"] += w * bs
            
                   
            # Per-region and per-hospital
            gt_r_3d = gt_r.reshape(bs, num_regions, pred_days)
            pred_r_3d = pred_r.reshape(bs, num_regions, pred_days)
            pred_h_3d = pred_h.reshape(bs, num_hospitals, pred_days)
            
            hage = hierarchical_aggregation_errors(pred_n, pred_r_3d, pred_h_3d, gt_n, gt_r_3d)
            for k, v in hage.items():
                test[f"hage_{k}"] += v * bs
            
            total_samples += bs
            
            
    # Averaging the metrics   
    results = {k: v / max(total_samples, 1) for k, v in test.items()}
    
    print('-----------------------------------------------------------------------------------------------------')
    print("Test Evaluation")
    print(f"Test Loss (hierarchical): {results['loss']:.6f}")
    print(f"  National  MAE {results['mae_n']:9.2f}  RMSE {results['rmse_n']:9.2f}  WAPE {results['wape_n'] * 100:6.2f}%")
    print(f"  Regional  MAE {results['mae_r']:9.2f}  RMSE {results['rmse_r']:9.2f}  WAPE {results['wape_r'] * 100:6.2f}%")
    print(f"  Hospital  MAE {results['mae_h']:9.2f}  RMSE {results['rmse_h']:9.2f}  WAPE {results['wape_h'] * 100:6.2f}%")

    print("  HAgE (prediction-side / vs ground truth)")
    print(f"    hospital -> regional  {results['hage_hosp_to_reg'] * 100:6.2f}%  /  {results['hage_hosp_to_reg_true'] * 100:6.2f}%")
    print(f"    hospital -> national  {results['hage_hosp_to_nat'] * 100:6.2f}%  /  {results['hage_hosp_to_nat_true'] * 100:6.2f}%")
    print(f"    regional -> national  {results['hage_reg_to_nat'] * 100:6.2f}%  /  {results['hage_reg_to_nat_true'] * 100:6.2f}%")
    print('-----------------------------------------------------------------------------------------------------')
    
    

if __name__ == "__main__":
    main()