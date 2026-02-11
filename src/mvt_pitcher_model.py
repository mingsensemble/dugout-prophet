from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MVTPitcherDataset(Dataset):
    def __init__(self, data, features, target, player_col='IDfg', season_col='Season', 
                 nlookbacks=5, min_qual_ip=50):
        
        self.data = data.copy()
        self.features = features
        self.player_col = player_col
        self.season_col = season_col
        self.nlookbacks = nlookbacks
        self.min_qual_ip = min_qual_ip
        self.target = target
        
        self.sequences, self.targets_features, self.ult_target, self.metadata_df = self._create_sequences()   

    def _create_sequences(self):
        features = self.features
        target = self.target
        data = self.data

        # Ensure data has the necessary columns
        required_cols = features + [target, self.player_col, self.season_col]
        data_processed = data.reindex(columns=[c for c in data.columns if c in required_cols]).fillna(0)

        # Group by player
        grouped = data_processed.groupby(self.player_col)

        sequences = []
        targets_features = []  # All feature targets - shape (num_sequences, len(features))
        ult_targets = []  # WAR targets
        metadata = []  # Track player_id, season for each row

        for player_id, player_data in grouped:
            # Sort by season
            player_data = player_data.sort_values(self.season_col).reset_index(drop=True)

            # Need at least 2 years: nlookback periods + 1 for target
            if len(player_data) >= 2:
                for i in range(len(player_data) - 1):
                    # Check qualification: target season must have at least min_qual_ip innings pitched
                    target_ip = player_data.iloc[i + 1]['IP']
                    if target_ip < self.min_qual_ip:
                        continue  # Skip this record if it doesn't meet qualification
                    
                    # lookback, padded with 0s if less
                    seq_length = min(self.nlookbacks, i + 1)
                    seq_data = np.zeros((self.nlookbacks, len(features)))

                    # Select the rows for this sequence
                    seq_rows = player_data.iloc[i - seq_length + 1:i + 1]

                    # Reindex columns to `features` to ensure consistent order
                    vals = seq_rows.reindex(columns=features).fillna(0).values
                    if vals.ndim == 1:
                        vals = vals.reshape(1, -1)

                    # Safety: if column count mismatches, trim or pad
                    if vals.shape[1] != len(features):
                        if vals.shape[1] > len(features):
                            vals = vals[:, :len(features)]
                        else:
                            pad = np.zeros((vals.shape[0], len(features) - vals.shape[1]))
                            vals = np.hstack([vals, pad])

                    # Place at the end of seq_data (right-align)
                    seq_data[self.nlookbacks - seq_length:] = vals

                    # Target: All features and WAR of next year
                    target_season = player_data.iloc[i + 1][self.season_col]
                    ult_target = player_data.iloc[i + 1][target]
                    
                    # Extract all feature targets in a loop
                    feature_targets = []
                    for feature_name in features:
                        feature_value = player_data.iloc[i + 1][feature_name]
                        feature_targets.append(feature_value)

                    sequences.append(seq_data)
                    targets_features.append(feature_targets)
                    ult_targets.append(ult_target)
                    metadata.append({
                        'player_id': player_id,
                        'current_season': int(player_data.iloc[i][self.season_col]),
                        'target_season': int(target_season),
                        'target_ip': float(target_ip),
                    })

        # Convert to numpy arrays and torch tensors
        sequences = np.array(sequences)  # Shape: (num_sequences, nlookback, len(features))
        targets_features = np.array(targets_features)  # Shape: (num_sequences, len(features))
        ult_targets = np.array(ult_targets)  # Shape: (num_sequences,) - WAR targets
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets_features = torch.tensor(targets_features, dtype=torch.float32)
        ult_targets = torch.tensor(ult_targets, dtype=torch.float32)

        metadata_df = pd.DataFrame(metadata)
        metadata_df = metadata_df.set_index(['player_id', 'target_season'])

        return sequences, targets_features, ult_targets, metadata_df
    


class MVTPitcherPredictionDataSet(Dataset):

    def __init__(self, data, pred_season, features, nlookbacks=5, player_col='IDfg', season_col='Season'):
        self.data = data.copy()
        self.pred_season = pred_season
        self.player_col = player_col
        self.season_col = season_col
        self.nlookbacks = nlookbacks
        self.features = features
        self.pred_sequences, self.pred_metadata = self._create_prediction_sequences()
    
    def _create_prediction_sequences(self):

        # Create prediction set for 2025 season
        pred_season = self.pred_season
        nlookbacks = self.nlookbacks

        pred_sequences = []
        pred_metadata = []

        data = self.data
        required_cols = self.features + [self.player_col, self.season_col]

        data_processed = data.reindex(columns=[c for c in data.columns if c in required_cols]).fillna(0)
        grouped = data_processed.groupby('IDfg')


        for player_id, player_data in grouped:
            # Sort by season
            player_data = player_data.sort_values('Season').reset_index(drop=True)
            
            # Check if player has data for pred_season
            if pred_season in player_data['Season'].values:
                # Find the index of pred_season
                pred_idx = player_data[player_data['Season'] == pred_season].index[0]
                
                # Build sequence using data up to pred_season
                seq_length = min(nlookbacks, pred_idx + 1)
                seq_data = np.zeros((nlookbacks, len(self.features)))
                
                # Select rows from pred_season going back nlookback periods
                seq_rows = player_data.iloc[pred_idx - seq_length + 1:pred_idx + 1]
                
                # Reindex to ensure consistent column order
                vals = seq_rows.reindex(columns=self.features).fillna(0).values
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)
                
                # Handle column mismatches
                if vals.shape[1] != len(self.features):
                    if vals.shape[1] > len(self.features):
                        vals = vals[:, :len(self.features)]
                    else:
                        pad = np.zeros((vals.shape[0], len(self.features) - vals.shape[1]))
                        vals = np.hstack([vals, pad])
                
                # Place at end of seq_data (right-align)
                seq_data[nlookbacks - seq_length:] = vals
                
                pred_sequences.append(seq_data)
                pred_metadata.append({
                    'player_id': player_id,
                    'prediction_season': pred_season,
                })

        # Convert to torch tensors
        pred_sequences = torch.tensor(np.array(pred_sequences), dtype=torch.float32)
        pred_metadata_df = pd.DataFrame(pred_metadata).set_index('player_id')
        return pred_sequences, pred_metadata_df

class MVTPitcherModel(nn.Module):
    def __init__(self, input_dim, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, seq_len=5):
        """
        Flexible Transformer model for predicting next-year values of all features and WAR.
        WAR prediction depends on predictions of all other features.
        
        Args:
            input_dim: Number of features (pitcher statistics)
            num_features: Number of features to predict (length of features list)
            d_model: Dimension of the transformer embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            seq_len: Length of input sequences (lookback window)
        """
        super(MVTPitcherModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Linear projection from input features to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction heads for each feature (independent)
        self.feature_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * seq_len, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
            for _ in range(num_features)
        ])
        
        # WAR prediction head (depends on transformer output + all predicted features)
        # Takes d_model * seq_len features + num_features for all feature predictions
        self.war_head = nn.Sequential(
            nn.Linear(d_model * seq_len + num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """Forward pass through transformer model.
        
        Returns:
            feature_preds: Predicted next-year values for all features (batch_size, num_features)
            war_pred: Predicted next-year WAR (batch_size,)
        """
        # Transformer encoding
        x_encoded = self.input_projection(x)
        x_encoded = x_encoded + self.positional_encoding
        x_encoded = self.transformer_encoder(x_encoded)
        x_flat = x_encoded.reshape(x_encoded.size(0), -1)
        
        # Predict all features independently
        feature_preds_list = []
        for head in self.feature_heads:
            pred = head(x_flat).squeeze(-1)
            feature_preds_list.append(pred)
        
        feature_preds = torch.stack(feature_preds_list, dim=1)  # (batch_size, num_features)
        
        # Predict WAR using transformer output and all predicted features
        war_input = torch.cat([x_flat, feature_preds], dim=1)
        war_pred = self.war_head(war_input).squeeze(-1)
        
        return feature_preds, war_pred