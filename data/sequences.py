"""Turning processed level frames into supervised (encoder, horizon) windows.

Splits are strictly chronological -- fixed cut-off dates, no shuffling -- so no
future information can reach a training window. Scalers are fitted on the
training split only.
"""

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Temporal splitting and sequence creation
# -----------------------------------------------------------------------------

def create_data_splits(data, train_threshold, val_threshold):

    train_data = data[data['time'] <= train_threshold]
    val_data = data[(data['time'] > train_threshold) & (data['time'] <= val_threshold)]
    test_data = data[data['time'] > val_threshold]

    return train_data, val_data, test_data


def create_sequences(data, input_days, pred_days, level, static_cols, categorical_cols, real_cols, log_target=True):

    if level == 'National':
        X, y = [], []
        feature_cols = list(static_cols) + list(real_cols) + list(categorical_cols)
        data_ord = data[feature_cols]  

        for i in range(len(data_ord) - input_days - pred_days + 1):
            X_win = data_ord.iloc[i:(i + input_days + pred_days)].values.copy()
            X.append(X_win)
            y_raw = data_ord.iloc[(i + input_days):(i + input_days + pred_days)]['M1'].values
            y.append(np.log1p(y_raw) if log_target else y_raw)
            
        X = np.array(X) 
        y = np.array(y)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    
    else: 
        # For regional or hospital level, use a dictionary to hold data for each entity
        group_col = 'RHA' if level == 'Regional' else 'D0'
        entities = data[group_col].unique()
        
        cat_vars = list(categorical_cols)
        
        X_entity, y_entity, X_future = [], [], []
        for entity in entities:
            subset = data[data[group_col] == entity].sort_values("time")
            X_data, y_data, X_data_future = [], [], []
            for i in range(len(subset) - input_days - pred_days + 1):
                X_data.append(subset.iloc[i:(i + input_days)].values)
                X_data_future.append(subset.iloc[(i + input_days):(i + input_days + pred_days)][cat_vars].values)
                y_raw = subset.iloc[i + input_days:(i + input_days + pred_days)]['M1'].values
                y_data.append(np.log1p(y_raw) if log_target else y_raw)
            X_future.append(X_data_future)
            X_entity.append(X_data)
            y_entity.append(y_data)
            
        X_entity = np.array(X_entity) 
        entity, samples, days, features_per_entity = X_entity.shape
        X = X_entity.transpose(1,2,0,3).reshape(samples, days, entity * features_per_entity) 
        
        X_future = np.array(X_future) 
        entity, samples, days, features_per_entity = X_future.shape
        X_fut = X_future.transpose(1,0,2,3).reshape(samples, entity * days, features_per_entity) 
        
        y_entity = np.array(y_entity)
        entity, samples, days = y_entity.shape
        y = y_entity.transpose(1,0,2).reshape(samples, entity * days) 
        
        return np.asarray(X, dtype=np.float32), np.asarray(X_fut, dtype=np.float32), np.asarray(y, dtype=np.float32)

    
# -----------------------------------------------------------------------------
# PyTorch datasets
# -----------------------------------------------------------------------------
class TFTDataset(torch.utils.data.Dataset):
    def __init__(self, X_full, y, num_ids):
        self.X = X_full if torch.is_tensor(X_full) else torch.tensor(X_full, dtype=torch.float32)
        self.y = y if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32)
        self.num_ids = int(num_ids)

    def __len__(self):
        return self.X.shape[0] # Folds * Entity

    def __getitem__(self, idx):
    
        x_seq = self.X[idx]  # (seq_len, F_total)
        y_seq = self.y[idx]  # (pred_len,)
        
        identifier = x_seq[:, :self.num_ids].long() # (Folds * Entity, 1, num_ids)
        inputs = x_seq[:, self.num_ids:] # (Folds * Entity, Seq, F)
        
        x = {
            "inputs": inputs,          
            "identifier": identifier        
        }
        return x, y_seq               # y: (Folds * Entity, pred_len)

class HierarchicalMixedDataset(torch.utils.data.Dataset):
    def __init__(self, national_dataset, regional_dataset, hospital_dataset):
        n = len(national_dataset)
        if len(regional_dataset) != n or len(hospital_dataset) != n:
            raise ValueError(
                f"All datasets must have the same length. "
                f"Got national={len(national_dataset)}, regional={len(regional_dataset)}, hospital={len(hospital_dataset)}"
            )

        self.national_dataset = national_dataset
        self.regional_dataset = regional_dataset
        self.hospital_dataset = hospital_dataset

    def __len__(self):
        return len(self.national_dataset)

    def __getitem__(self, idx):
        pt_x, pt_y = self.national_dataset[idx]
        r_x, r_fut, r_y = self.regional_dataset[idx]
        h_x, h_fut, h_y = self.hospital_dataset[idx]

        return pt_x, pt_y, r_x, r_fut, r_y, h_x, h_fut, h_y