import pandas as pd
from datetime import datetime
import os
from pybaseballstats.statcast import pitch_by_pitch_data
from pybaseballstats.utils.retrosheet_utils import _get_people_data
import polars as pl
from constants import CURRENT_YEAR

MANUAL_ID_TO_NAME = {
    700250: "Ben Rice",
    686948: "Drake Baldwin",
    701398: "Sal Stewart",
    693307: "Dillon Dingler",
    669257: "Will Smith",
    665742: "Juan Soto",
    701762: "Nick Kurtz",
    605137: "Josh Bell",
    695578: "James Wood",
    666182: "Bo Bichette",
    679529: "Spencer Torkelson",
    621493: "Taylor Ward",
    608070: "José Ramírez",
    681624: "Andy Pages",
    701358: "Cam Smith",
    800050: "Chase Delauter",
    665052: "Griffin Conine",
    687221: "Dalton Rushing",
    624413: "Pete Alonso",
}

CACHE_DIR = "data"


def _fetch_and_cache(year, start_dt, end_dt, cache_path):
    result = pitch_by_pitch_data(start_date=start_dt, end_date=end_dt, force_collect=True)
    df = result.to_pandas() if result is not None else pd.DataFrame()
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
    people = _get_people_data()
    ids_pl = pl.Series("key_mlbam", [int(x) for x in batter_ids])
    id_map = (
        people.filter(pl.col("key_mlbam").is_in(ids_pl))
        .select(["key_mlbam", "name_first", "name_last"])
        .to_pandas()
    )
    id_map['player_name'] = (
        id_map['name_first'] + ' ' + id_map['name_last']
    ).str.title()
    id_map = id_map.rename(columns={'key_mlbam': 'batter'})

    raw = raw.merge(id_map[['batter', 'player_name']], on='batter', how='left')
    raw['player_name'] = raw['player_name'].fillna(raw['batter'].astype(str))

    # patch any IDs missing from the pybaseballstats lookup
    missing = raw['player_name'] == raw['batter'].astype(str)
    raw.loc[missing, 'player_name'] = raw.loc[missing, 'batter'].map(
        lambda bid: MANUAL_ID_TO_NAME.get(int(bid), str(bid))
    )

    return raw[['batter', 'player_name', 'game_date',
                'estimated_woba_using_speedangle', 'woba_value', 'days_ago']]
