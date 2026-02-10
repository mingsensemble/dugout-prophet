from pybaseball import statcast
import warnings
import pandas as pd
import numpy as np

# Suppress all FutureWarning messages from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_havaa(plate_z, release_pos_z, release_pos_y):
    values = np.degrees(
        np.arctan((plate_z - release_pos_z) / (0 - release_pos_y))
    )
    return values

def calculate_extensions(release_pos_y):
    return 60.5 - release_pos_y

def calculate_csw(description):
    
    called_strikes =np.sum(description == 'called_strike')
    whiffs = np.sum(description.isin(['swinging_strike', 'swinging_strike_blocked', 'swinging_pitchout', 'foul_tip']))
    csw_count = called_strikes +whiffs
    total_pitches = len(description)
    csw_percent = (csw_count / total_pitches) * 100 if total_pitches > 0 else 0
    return csw_percent

def calculate_fip(HR, BB, HBP, K, IP):
    FIP_constant = 3.1
    val = ((13 * HR) + (3 * (BB + HBP)) - (2 * K)) / IP + FIP_constant
    return val

def calculate_zone(zone):
    pitches_in_zone = zone.between(1,9).sum()
    total_pitches = len(zone)
    zone_percent = (pitches_in_zone / total_pitches) * 100 if total_pitches > 0 else 0
    return zone_percent    
    
def generate_pitcher_core_perf_trainingset(start_dt, end_dt, for_training = True, groupby = ['pitcher']):
    # 1. Load data from statcast
    df = statcast(start_dt, end_dt)
    id_mapping = df[['pitcher','player_name']].drop_duplicates()
    pt_mapping = df[['pitcher','p_throws']].drop_duplicates()
    # 2. Filter out only pitcher events (where 'outs_when_up' is not null)
    df = df[df['pitcher'].notnull()]
    df.loc[df['pitch_type'].isin(['EP', 'FA', 'SV', 'SC', 'KN']), 'pitch_type'] = 'other'
    df_ip = df.groupby(groupby+['batter', 'game_pk', 'inning']).first().groupby(groupby).agg(
        IP=('outs_when_up', lambda x:np.sum(x)/3),
    )
    game_by_p = df.groupby(groupby).agg(
        HR=('events', lambda x: (x == 'home_run').sum()),
        BB=('events', lambda x: (x == 'walk').sum()),
        HBP=('events', lambda x: (x == 'hit_by_pitch').sum()),
        SO=('events', lambda x: (x == 'strikeout').sum()),
        FB=('bb_type', lambda x: (x == 'fly_ball').sum())
    ).join(
        df_ip,
        how = 'left'
    ).fillna({'IP': 0}).reset_index()
    
    league_hr_fb = 0.106
    constant = 3.2
    game_by_p['expected_HR'] = game_by_p['FB'] * league_hr_fb
    game_by_p['xFIP'] = ((13 * game_by_p['expected_HR'] + 3 * (game_by_p['BB'] + game_by_p['HBP']) - 2 * game_by_p['SO']) / game_by_p['IP']) + constant
    game_by_p['FIP'] = ((13 * game_by_p['HR'] + 3 * (game_by_p['BB'] + game_by_p['HBP']) - 2 * game_by_p['SO']) / game_by_p['IP']) + constant

    df['havaa'] = calculate_havaa(df['plate_z'], df['release_pos_z'], df['release_pos_y'])
    df['exts'] = calculate_extensions(df['release_pos_y'])
    
    sel_cols = ['havaa', 'exts', 'description','zone', 'woba_value', 'effective_speed', 'release_spin_rate', 'spin_axis']
    groupby_keys = groupby + ['stand', 'pitch_type',]
    
    features = df[groupby_keys+sel_cols].groupby(groupby_keys).agg(
        mean_havaa = ('havaa', 'mean'),
        max_havaa = ('havaa', 'max'),
        min_havaa = ('havaa', 'min'),
        mean_exts = ('exts', 'mean'),
        mean_velo = ('effective_speed', 'mean'),
        max_velo = ('effective_speed', 'max'),
        min_velo = ('effective_speed', 'min'),
        iqr_velo = ('effective_speed', lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        max_spin = ('release_spin_rate', 'max'),
        min_spin = ('release_spin_rate', 'min'),
        mean_spin = ('release_spin_rate', 'mean'),
        mean_spin_axis = ('spin_axis', 'mean'),
        zone_pct = ('zone', calculate_zone),
        csw = ('description', calculate_csw),
        mean_woba = ('woba_value', 'mean'),
        pitch_counts = ('havaa', 'size'),

        zone1=('zone', lambda x:np.sum(x==1)/len(x)*100),
        zone2=('zone', lambda x:np.sum(x==2)/len(x)*100),
        zone3=('zone', lambda x:np.sum(x==3)/len(x)*100),
        zone4=('zone', lambda x:np.sum(x==4)/len(x)*100),
        zone5=('zone', lambda x:np.sum(x==5)/len(x)*100),
        zone6=('zone', lambda x:np.sum(x==6)/len(x)*100),
        zone7=('zone', lambda x:np.sum(x==7)/len(x)*100),
        zone8=('zone', lambda x:np.sum(x==8)/len(x)*100),
        zone9=('zone', lambda x:np.sum(x==9)/len(x)*100),
        zone10=('zone', lambda x:np.sum(x==10)/len(x)*100),
        zone11=('zone', lambda x:np.sum(x==11)/len(x)*100),
        zone12=('zone', lambda x:np.sum(x==12)/len(x)*100),
        zone13=('zone', lambda x:np.sum(x==13)/len(x)*100),
        zone14=('zone', lambda x:np.sum(x==14)/len(x)*100),
    ).reset_index()
    if for_training:
        features = features.pivot(
            index = groupby,
            columns = ['stand', 'pitch_type',],
            values = ['mean_havaa', 'mean_exts', 'pitch_counts', 'zone_pct', 'csw', 'mean_woba', 'mean_velo', 'max_velo', 'min_velo']+[f'zone{i+1}' for i in range(14)],
        ).reset_index()
        features['pitch_counts'] = features['pitch_counts'].fillna(0)
        features.columns = ['_'.join(col).strip() for col in features.columns.values]
        features = features.rename(columns = {v+'__':v for v in groupby}).set_index(groupby)
        trainingset = features.join(game_by_p[['pitcher',  'xFIP', 'FIP']].set_index(['pitcher',]), how = 'inner').fillna(-99999)
        # convert pitch counts to percentage
        pitch_count_cols = [x for x in trainingset.columns if 'pitch_counts' in x]
        # aggregation is conditional on L/R
        trainingset = trainingset.copy()
        l_cols = [x for x in pitch_count_cols if '_L' in x]
        for c in l_cols:
            trainingset[f'{c}_pct'] = (trainingset[c]/(trainingset[l_cols].sum(axis = 1))*100).tolist()
            trainingset = trainingset.copy()
        r_cols = [x for x in pitch_count_cols if '_R' in x]
        for c in r_cols:
            trainingset[f'{c}_pct'] = (trainingset[c]/(trainingset[r_cols].sum(axis = 1))*100).tolist()
            trainingset = trainingset.copy()
        trainingset = trainingset.drop(columns = pitch_count_cols)
        trainingset = trainingset.copy()
        trainingset = trainingset[(trainingset['xFIP']<np.inf) & (trainingset['xFIP']>= 0)].join(
            pt_mapping.set_index('pitcher'),
            how = 'left'
        )
        return trainingset
    else:
        results = features.reset_index().drop('index', axis = 1).set_index('pitcher').join(
            id_mapping.set_index('pitcher'),
            how = 'left'
        )
        return results
