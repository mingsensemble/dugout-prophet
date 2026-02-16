from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MVTBatterDataset(Dataset):
    def __init__(self, data, features, player_col='IDfg', season_col='Season', 
                 nlookbacks=5, min_qual_pa=50):
        
        self.data = data.copy()
        self.features = features
        self.player_col = player_col
        self.season_col = season_col
        self.nlookbacks = nlookbacks
        self.min_qual_pa = min_qual_pa
        
        self.sequences, self.targets_features, self.metadata_df = self._create_sequences()   

    def _create_sequences(self):
        features = self.features
        data = self.data

        # Ensure data has the necessary columns
        required_cols = list(set(features + [self.player_col, self.season_col]) | set(['PA']))
        data_processed = data.reindex(columns=[c for c in data.columns if c in required_cols]).fillna(0)

        # Group by player
        grouped = data_processed.groupby(self.player_col)

        sequences = []
        targets_features = []  # All feature targets - shape (num_sequences, len(features))
        metadata = []  # Track player_id, season for each row

        for player_id, player_data in grouped:
            # Sort by season
            player_data = player_data.sort_values(self.season_col).reset_index(drop=True)

            # Need at least 2 years: nlookback periods + 1 for target
            if len(player_data) >= 2:
                for i in range(len(player_data) - 1):
                    # Check qualification: target season must have at least min_qual_pa plate appearances
                    target_pa = player_data.iloc[i + 1]['PA']
                    if target_pa < self.min_qual_pa:
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
                    
                    # Extract all feature targets in a loop
                    feature_targets = []
                    for feature_name in features:
                        feature_value = player_data.iloc[i + 1][feature_name]
                        feature_targets.append(feature_value)

                    sequences.append(seq_data)
                    targets_features.append(feature_targets)
                    metadata.append({
                        'player_id': player_id,
                        'current_season': int(player_data.iloc[i][self.season_col]),
                        'target_season': int(target_season),
                        'target_pa': float(target_pa),
                    })

        # Convert to numpy arrays and torch tensors
        sequences = np.array(sequences)  # Shape: (num_sequences, nlookback, len(features))
        targets_features = np.array(targets_features)  # Shape: (num_sequences, len(features))
        sequences = torch.tensor(sequences, dtype=torch.float32)
        targets_features = torch.tensor(targets_features, dtype=torch.float32)

        metadata_df = pd.DataFrame(metadata)
        metadata_df = metadata_df.set_index(['player_id', 'target_season'])

        return sequences, targets_features, metadata_df

class MVTBatterPredictionDataset(Dataset):

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

# Multivariate Transformer Model with Multihead Attention for next year feature prediction
class MVTBatterModel(nn.Module):
    def __init__(self, input_dim, num_features, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1, nlookbacks=5):
        """
        Multivariate Transformer model for predicting next year's features.
        
        Args:
            input_dim: Number of input features (equal to num_features)
            num_features: Number of features to predict for next year
            d_model: Dimension of the model (embedding dimension)
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward networks
            dropout: Dropout rate
            nlookbacks: Number of lookback years/seasons to use for prediction
        """
        super(MVTBatterModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_features = num_features
        self.d_model = d_model
        self.nlookbacks = nlookbacks
        
        # Input projection layer - transforms raw features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for sequence
        self.register_buffer('pos_encoding', self._generate_pos_encoding(d_model, nlookbacks))
        
        # Transformer encoder with multihead attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for feature prediction
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_features)
        )
    
    def _generate_pos_encoding(self, d_model, seq_len):
        """Generate positional encoding for transformer."""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
    
    def forward(self, x):
        """
        Forward pass for the transformer model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               Where seq_len is the number of historical seasons/years

        Returns:
            output: Predicted features for next year of shape (batch_size, num_features)
        """
        batch_size, seq_len, _ = x.shape

        # Project input features to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # Generate or retrieve positional encoding for current sequence length
        if seq_len > self.pos_encoding.size(1):
            # If input sequence is longer than stored encoding, generate new one
            pos_encoding = self._generate_pos_encoding(self.d_model, seq_len).to(x.device)
        else:
            # Use stored positional encoding
            pos_encoding = self.pos_encoding[:, :seq_len, :]

        # Add positional encoding
        x = x + pos_encoding  # (batch_size, seq_len, d_model)

        # Apply transformer encoder with multihead attention
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Use the last sequence element (most recent year) for prediction
        # Alternatively, could use mean pooling: transformer_out.mean(dim=1)
        last_output = transformer_out[:, -1, :]  # (batch_size, d_model)

        # Predict next year's features
        output = self.fc_out(last_output)  # (batch_size, num_features)

        return output


