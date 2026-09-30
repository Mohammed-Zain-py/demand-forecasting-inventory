import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DemandSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # take last sequence step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(1)

class PyTorchLSTMModel:
    def __init__(self, seq_len=28, hidden_size=64, num_layers=1, epochs=15, batch_size=256, lr=0.005):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.features = [
            'lag_1', 'lag_7', 'lag_14', 'lag_28',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_28',
            'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'is_weekend',
            'onpromotion', 'promotion_known', 'store_closed'
        ]
        # We rely on the lags to capture historical patterns without increasing dimensionality.
        
    def _create_sequences(self, df, is_train=True, history_df=None):
        X_seq = []
        y_seq = []
        
        # Determine history subset
        if history_df is not None:
            hist_df_subset = history_df.groupby('item_nbr', sort=False).tail(self.seq_len - 1)
        else:
            hist_df_subset = None
            
        # Prepare feature dfs
        feat_df = df[self.features].copy()
        for col in ['onpromotion', 'is_weekend', 'promotion_known', 'store_closed']:
            if col in feat_df.columns:
                feat_df[col] = feat_df[col].astype(float)
                
        if hist_df_subset is not None:
            hist_feat_df = hist_df_subset[self.features].copy()
            for col in ['onpromotion', 'is_weekend', 'promotion_known', 'store_closed']:
                if col in hist_feat_df.columns:
                    hist_feat_df[col] = hist_feat_df[col].astype(float)
                    
        # Scale features
        if is_train:
            scaled_feats = self.feature_scaler.fit_transform(feat_df)
        else:
            scaled_feats = self.feature_scaler.transform(feat_df)
            if hist_df_subset is not None:
                scaled_hist_feats = self.feature_scaler.transform(hist_feat_df)
            
        scaled_df = pd.DataFrame(scaled_feats, columns=self.features, index=df.index)
        scaled_df['item_nbr'] = df['item_nbr'].values
        
        if hist_df_subset is not None:
            hist_scaled_df = pd.DataFrame(scaled_hist_feats, columns=self.features, index=hist_df_subset.index)
            hist_scaled_df['item_nbr'] = hist_df_subset['item_nbr'].values
        
        # Scale targets
        if is_train and 'unit_sales' in df.columns:
            targets = self.target_scaler.fit_transform(df[['unit_sales']]).flatten()
        elif 'unit_sales' in df.columns:
            targets = self.target_scaler.transform(df[['unit_sales']]).flatten()
        else:
            targets = None
            
        scaled_df['target'] = targets if targets is not None else 0.0
        
        for item_nbr, group in scaled_df.groupby('item_nbr', sort=False):
            feat_vals = group[self.features].values
            targ_vals = group['target'].values
            
            if hist_df_subset is not None and item_nbr in hist_scaled_df['item_nbr'].values:
                hist_group = hist_scaled_df[hist_scaled_df['item_nbr'] == item_nbr]
                hist_feat_vals = hist_group[self.features].values
                
                pad_len = max(0, (self.seq_len - 1) - len(hist_feat_vals))
                if pad_len > 0:
                    hist_feat_vals = np.pad(hist_feat_vals, ((pad_len, 0), (0, 0)), mode='constant')
                padded_feats = np.vstack([hist_feat_vals, feat_vals])
            else:
                padded_feats = np.pad(feat_vals, ((self.seq_len - 1, 0), (0, 0)), mode='constant')
            
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(padded_feats, window_shape=self.seq_len, axis=0)
            windows = np.swapaxes(windows, 1, 2)
            
            X_seq.append(windows)
            if targets is not None:
                y_seq.append(targ_vals)
                
        X_seq = np.concatenate(X_seq, axis=0)
        if targets is not None:
            y_seq = np.concatenate(y_seq, axis=0)
            return X_seq, y_seq
        return X_seq, None

    def train(self, df_train):
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.history_df = df_train.copy()
        
        self.model = LSTMNet(input_size=len(self.features), 
                             hidden_size=self.hidden_size, 
                             num_layers=self.num_layers)
        
        X, y = self._create_sequences(df_train, is_train=True)
        
        dataset = DemandSequenceDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
    def predict(self, df_test, history_df=None):
        self.model.eval()
        if history_df is None and hasattr(self, 'history_df'):
            history_df = self.history_df
            
        X, _ = self._create_sequences(df_test, is_train=False, history_df=history_df)
        dataset = DemandSequenceDataset(X, None)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_X in dataloader:
                outputs = self.model(batch_X)
                preds.extend(outputs.numpy())
                
        preds = np.array(preds).reshape(-1, 1)
        preds = self.target_scaler.inverse_transform(preds).flatten()
        return np.clip(preds, 0, None)
