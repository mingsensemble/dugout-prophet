from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class CNNPitcherDataset(Dataset):
    def __init__(self, data, features, target, player_col='IDfg', season_col='Season', 
                 nlookbacks=5, min_qual_ip=50):
        """
        Args:
            data: DataFrame with batting statistics
            features: List of feature column names
            player_col: Column name for player ID
            season_col: Column name for season year
            nlookbacks: Number of lookback years for sequences
            min_qual_ip: Minimum innings pitched to include a player-season
        """
        self.data = data.copy()
        self.features = features
        self.player_col = player_col
        self.season_col = season_col
        self.nlookbacks = nlookbacks
        self.min_qual_ip = min_qual_ip
        self.target = target
        
        self.sequences, self.target = self._create_sequences()        
    def _create_sequences(self):
        """Create training sequences from player-season data."""

        # Ensure data has the necessary columns
        required_cols = self.features + [self.target, self.player_col, self.season_col]
        data_processed = self.data.reindex(columns=[c for c in self.data.columns if c in required_cols]).fillna(0)

        # Group by player
        grouped = data_processed.groupby(self.player_col)

        sequences = []
        targets = []
        metadata = []  # Track player_id, season for each row

        nlookbacks = self.nlookbacks
        features = self.features

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
                    seq_length = min(nlookbacks, i + 1)
                    seq_data = np.zeros((nlookbacks, len(features)))

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
                    seq_data[nlookbacks - seq_length:] = vals

                    # Target: WAR of next year
                    target_season = player_data.iloc[i + 1][self.season_col]
                    target_war = player_data.iloc[i + 1][self.target]

                    sequences.append(seq_data)
                    targets.append(target_war)
                    metadata.append({
                        'player_id': player_id,
                        'current_season': int(player_data.iloc[i][self.season_col]),
                        'target_season': int(target_season),
                        'target_ip': float(target_ip),
                    })

        # Convert to numpy arrays and torch tensors
        sequences = np.array(sequences)  # Shape: (num_sequences, nlookbacks, len(features))
        targets = np.array(targets)      # Shape: (num_sequences,)
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.metadata = metadata  # Keep metadata as a list of dicts for reference
        
        return sequences, targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
class CNNPitcherPredictionDataset:
    def __init__(self, data, pred_season, features, nlookbacks=5, player_col='IDfg', season_col='Season'):
        self.data = data.copy()
        self.pred_season = pred_season
        self.player_col = player_col
        self.season_col = season_col
        self.nlookback = nlookbacks
        self.features = features
        self.sequences, self.metadata = self._create_prediction_sequences()
        
    def _create_prediction_sequences(self):
        """Create sequences for prediction based on data up to pred_season."""
        # Group by player
        grouped = self.data.groupby(self.player_col)
        # Create prediction set for 2025 season
        pred_season = self.pred_season
        features = self.features

        pred_sequences = []
        pred_metadata = []

        for player_id, player_data in grouped:
            # Sort by season
            player_data = player_data.sort_values(self.season_col).reset_index(drop=True)
            
            # Check if player has data for pred_season
            if pred_season in player_data[self.season_col].values:
                # Find the index of pred_season
                pred_idx = player_data[player_data[self.season_col] == pred_season].index[0]
                
                # Build sequence using data up to pred_season
                seq_length = min(self.nlookback, pred_idx + 1)
                seq_data = np.zeros((self.nlookback, len(self.features)))
                
                # Select rows from pred_season going back nlookback periods
                seq_rows = player_data.iloc[pred_idx - seq_length + 1:pred_idx + 1]
                
                # Reindex to ensure consistent column order
                vals = seq_rows.reindex(columns=features).fillna(0).values
                if vals.ndim == 1:
                    vals = vals.reshape(1, -1)
                
                # Handle column mismatches
                if vals.shape[1] != len(features):
                    if vals.shape[1] > len(features):
                        vals = vals[:, :len(features)]
                    else:
                        pad = np.zeros((vals.shape[0], len(features) - vals.shape[1]))
                        vals = np.hstack([vals, pad])
                
                # Place at end of seq_data (right-align)
                seq_data[self.nlookback - seq_length:] = vals
                
                pred_sequences.append(seq_data)
                pred_metadata.append({
                    'player_id': player_id,
                    'prediction_season': pred_season,
                })

        # Convert to torch tensors
        pred_sequences = torch.tensor(np.array(pred_sequences), dtype=torch.float32)
        pred_metadata_df = pd.DataFrame(pred_metadata).set_index('player_id')
        return pred_sequences, pred_metadata_df

class CNNPitcherModel(nn.Module):
    def __init__(self, input_channels=6, seq_length=3, num_classes=1):
        super(CNNPitcherModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Use adaptive pooling to ensure output size
        self.dropout = nn.Dropout(0.4)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1)  # kernel=1 to avoid size issues
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Use adaptive pooling to ensure output size
        self.dropout = nn.Dropout(0.4)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (batch, seq_length=3, features=6)
        # Permute to (batch, features=6, seq_length=3) for Conv1d
        x = x.permute(0, 2, 1)

        x = torch.relu(self.conv1(x))  # (batch, 32, 2)
        x = self.pool(x)  # (batch, 32, 1)
        x = self.bn1(x)
        x = self.dropout(x)
   
        
        x = torch.relu(self.conv2(x))  # (batch, 64, 2)
        x = self.pool(x)  # (batch, 64, 1)
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = self.flatten(x)  # (batch, 64)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.fc2(x)
        return x
