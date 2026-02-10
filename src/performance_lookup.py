import pandas as pd
import numpy as np
from pybaseball import statcast

class performance_lookup:
    def __init__(self, data, query):
        self.query = query
        self.data  = data.query(query)
    def lookup(self, name):
        one_player = self.data.loc[self.data['player_name'] == name, ['player_name', 'pred_FIP','FIP','pred_xFIP', 'xFIP', 'IP']]
        if one_player.shape[0] == 0:
            one_player = self.data[self.data['player_name'].str.contains(name)]
            name = one_player['player_name'].values[0]

        print(f'{name} has pitched {one_player['IP'].values[0]} innings:')
        for col in ['pred_FIP','FIP','pred_xFIP', 'xFIP',]:
            val = one_player[col].values[0]
            pctl = (self.data[col] > val).mean()*100
            print(f'top {pctl:.1f}th percentile in {col}')

        print(f'among pitchers with {self.query}')
    def leaderboard(self, top_n, orderby = 'pred_FIP'):
        output = self.data.loc[:, ['player_name', 'pred_FIP','FIP','pred_xFIP', 'xFIP', 'IP', ]].sort_values(orderby).head(top_n)
        return output

def calculate_pitching_stats(start_date, end_date):
    df = statcast(start_date, end_date)
    stat_df = df[df['inning']==1].groupby(['game_pk', 'inning_topbot']).first()[['player_name', 'pitcher', 'events']].groupby(['player_name', 'pitcher']).agg(
        GS=('events', 'size')
    ).join(
        df.groupby(['player_name', 'pitcher', 'batter', 'game_pk', 'inning']).first().groupby(['player_name', 'pitcher',]).agg(
            IP=('outs_when_up', lambda x:np.sum(x)/3),
        ),
        how = 'right'
    ).fillna({'GS': 0})
    return stat_df