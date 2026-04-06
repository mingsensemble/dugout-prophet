import pybaseball as pb
import pandas as pd
from datetime import datetime
import os
from constants import CURRENT_YEAR

CACHE_DIR = "data"


def _fetch_and_cache(year, start_dt, end_dt, cache_path):
    df = pb.statcast(start_dt=start_dt, end_dt=end_dt)
    df.to_parquet(cache_path, index=False)
    return df


def pull_or_load(year: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/statcast_{year}.parquet"

    is_current_year = (year == CURRENT_YEAR)
    cache_exists = os.path.exists(cache_path)

    if not is_current_year and cache_exists:
        df = pd.read_parquet(cache_path)
    elif is_current_year and cache_exists:
        last_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - last_modified).total_seconds() < 86400:
            df = pd.read_parquet(cache_path)
        else:
            df = _fetch_and_cache(
                year,
                start_dt=f'{year}-03-01',
                end_dt=datetime.now().strftime('%Y-%m-%d'),
                cache_path=cache_path
            )
    else:
        end_dt = datetime.now().strftime('%Y-%m-%d') if is_current_year else f'{year}-10-31'
        df = _fetch_and_cache(
            year,
            start_dt=f'{year}-03-01',
            end_dt=end_dt,
            cache_path=cache_path
        )

    return df


def pull_statcast_data(years: list) -> pd.DataFrame:
    dfs = [pull_or_load(year) for year in years]
    raw = pd.concat(dfs, ignore_index=True)

    raw = raw[raw['events'].notnull()]
    raw = raw[raw['estimated_woba_using_speedangle'].notnull()]

    raw['game_date'] = pd.to_datetime(raw['game_date'])
    today = datetime.now().date()
    raw['days_ago'] = raw['game_date'].dt.date.apply(lambda d: (today - d).days)

    # drop pitcher name, look up batter names by MLBAM ID
    raw = raw.drop(columns=['player_name'], errors='ignore')
    batter_ids = raw['batter'].dropna().unique().tolist()
    id_map = pb.playerid_reverse_lookup(batter_ids, key_type='mlbam')[
        ['key_mlbam', 'name_first', 'name_last']
    ]
    id_map['player_name'] = (
        id_map['name_first'] + ' ' + id_map['name_last']
    ).str.title()  # ← capitalization fix here
    id_map = id_map.rename(columns={'key_mlbam': 'batter'})

    raw = raw.merge(id_map[['batter', 'player_name']], on='batter', how='left')
    raw['player_name'] = raw['player_name'].fillna(raw['batter'].astype(str))

    # return raw[['batter', 'player_name', 'game_date',
    #             'estimated_woba_using_speedangle', 'days_ago']]
    # in pull_statcast_data return statement — add woba_value
    return raw[['batter', 'player_name', 'game_date',
                'estimated_woba_using_speedangle', 'woba_value', 'days_ago']]