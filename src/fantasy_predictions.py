import logging
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import MCMC, NUTS, Predictive

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.9.1')

import sys

pd.options.mode.chained_assignment = None

# create a function
# filename: fantasy_predictions.py

class fantasy_predictions:
    def __init__(self, trn_data, test_data, device='cpu'):
        self.trn_data = trn_data
        self.test_data = test_data
        # Normalize device selection to torch.device
        if isinstance(device, str):
            if device.lower() in ('cuda', 'gpu'):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif device.lower() in ('mps', 'metal'):
                self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = device

    def _prep_data(self):
        stats_cols = [
            'PA','AB','BB','SO','BIP','H','1B','2B','3B','HR','ITP','Age',
        ]
        self.trn_data.loc[:, 'BIP'] = (self.trn_data['AB'] - self.trn_data['SO']).values
        self.trn_data.loc[:, 'OBB'] = (self.trn_data['BB'] + self.trn_data['IBB'] + self.trn_data['HBP']).values
        self.trn_data.loc[:, 'ITP'] = (self.trn_data['H'] + self.trn_data['HR']).values

        self.trn_data = self.trn_data[['Name', 'IDfg'] + stats_cols]
        self.trn_data = self.trn_data.set_index('IDfg').join(
            self.test_data[['IDfg']].set_index('IDfg'), how='inner'
        ).reset_index()

        self.test_data = self.test_data[['Name','IDfg','PA','Age']].set_index('IDfg').join(
            self.trn_data[['IDfg']].set_index('IDfg'), how='inner'
        ).reset_index()
        return None

    @staticmethod
    def logit(p):
        return np.log(p / (1 - p))

    @staticmethod
    def model(trials, age, player, prior_mu, prior_sigma, hits=None):
        # Ensure constants live on the same device as inputs
        device = trials.device
        zero = torch.tensor(0.0, device=device)
        one = torch.tensor(1.0, device=device)
        two = torch.tensor(2.0, device = device)
        # four = torch.tensor(4.0, device=device)

        # player plate with per-player location
        with pyro.plate('player', len(torch.unique(player))):
            player_loc = pyro.sample("loc", dist.Normal(prior_mu, prior_sigma))

        beta1 = pyro.sample('beta1', dist.Normal(zero, one))
        beta2 = pyro.sample('beta2', dist.Normal(zero, one))
        scale = pyro.sample('scale', dist.HalfNormal(two))

        mu = player_loc[player] + beta1 * age + beta2 * (age ** 2)
        alpha = pyro.sample("alpha", dist.Normal(mu, one))

        # Binomial total_count must be non-negative; prefer integer dtype
        return pyro.sample("obs", dist.Binomial(total_count=trials.to(dtype=torch.int64), logits=alpha), obs=hits)

    def create_torch(self, trial, hit):
        # Float32 tensors on the chosen device
        trial_torch = torch.tensor(self.trn_data[trial].values, dtype=torch.float32, device=self.device)
        hit_torch = torch.tensor(self.trn_data[hit].values, dtype=torch.float32, device=self.device)
        age_torch = torch.tensor(self.trn_data['Age'].values - 19, dtype=torch.float32, device=self.device)

        # Index tensor should be long for indexing
        player_torch = torch.tensor(self.trn_data['IDfg'].values, dtype=torch.long, device=self.device)
        unique_ids, player = torch.unique(player_torch, return_inverse=True)

        prior_mu = torch.tensor(
            self.logit(self.trn_data[hit].sum() / self.trn_data[trial].sum()),
            dtype=torch.float32, device=self.device
        )
        prior_sigma = torch.tensor(0.01, dtype=torch.float32, device=self.device)

        output = {
            'trials': trial_torch,
            'age': age_torch,
            'player': player,          # long dtype on device
            'prior_mu': prior_mu,
            'prior_sigma': prior_sigma,
            'hits': hit_torch,
        }
        return output

    def train(self, num_samples=500, warmup_steps=200, num_chains=1):
        self._prep_data()
        self.model_dict = {}
        self.model_config = {
            'BB': ['PA', 'BB'],
            'SO': ['AB', 'SO'],
            'H': ['BIP', 'H'],
            'HR': ['H', 'HR'],
            '1B': ['ITP', '1B'],
            '3B': ['ITP', '3B'],
            '2B': ['ITP', '2B'],
        }

        for item in self.model_config.items():
            print(f'train {item[0]} Model')
            bino_model = self.model
            input_torch = self.create_torch(*item[1])

            nuts_kernel = NUTS(bino_model)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains)
            mcmc.run(**input_torch)  # runs on device because inputs are on device

            self.model_dict.update({
                f'{item[0]}': {
                    'model': bino_model,
                    'mcmc': mcmc,
                    'prior_mu': input_torch['prior_mu'],
                    'prior_sigma': input_torch['prior_sigma'],
                }
            })
        return None

    def _predict(self, model_id, trials, metric):
        age_torch = torch.tensor(self.test_data['Age'].values, dtype=torch.float32, device=self.device)

        # Use integer dtype for total_count in Binomial
        trials_counts = np.ceil(self.test_data[trials].values).astype(np.int64)
        trials_torch = torch.tensor(trials_counts, dtype=torch.int64, device=self.device)

        player_torch = torch.tensor(self.test_data['IDfg'].values, dtype=torch.long, device=self.device)
        unique_ids, player = torch.unique(player_torch, return_inverse=True)

        predictive = Predictive(self.model_dict[model_id]['model'], self.model_dict[model_id]['mcmc'].get_samples())

        pred = predictive(
            trials_torch,
            age_torch,
            player,
            self.model_dict[model_id]['prior_mu'],
            self.model_dict[model_id]['prior_sigma']
        )['obs']

        if metric == 'mean':
            self.test_data[model_id] = pred.mean(axis=0).cpu().numpy()
        else:
            self.test_data[model_id] = pred.quantile(float(metric), axis=0).cpu().numpy()
        return None

    def predict(self, metric='mean'):
        print('predict BB')
        self._predict('BB', 'PA', metric)
        self.test_data['AB'] = self.test_data['PA'] - self.test_data['BB']

        print('predict SO')
        self._predict('SO', 'AB', metric)
        self.test_data['BIP'] = self.test_data['AB'] - self.test_data['SO']

        print('predict H')
        self._predict('H', 'BIP', metric)

        print('predict HR')
        self._predict('HR', 'H', metric)
        self.test_data['ITP'] = self.test_data['H'] - self.test_data['HR']

        print('predict 3B')
        self._predict('3B', 'ITP', metric)

        print('predict 2B')
        self._predict('2B', 'ITP', metric)

        print('predict 1B')
        self._predict('1B', 'ITP', metric)
        # self.test_data['1B'] = self.test_data['ITP'] - self.test_data['2B'] - self.test_data['3B']

        self.test_data = self.test_data[['Name','Age','PA','AB','BB','SO','1B','2B','3B','HR']]
        return None
