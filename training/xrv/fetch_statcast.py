import time
import warnings
import calendar
from datetime import date

import pandas as pd
from pybaseball import statcast


def monthly_chunks(start: str, end: str) -> list[tuple[str, str]]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)

    chunks = []
    year, month = s.year, s.month
    while date(year, month, 1) <= e:
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
        chunks.append((
            max(first, s).isoformat(),
            min(last, e).isoformat()
        ))
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return chunks


def fetch_statcast(start: str, end: str, retries: int = 3, retry_delay: float = 10.0) -> pd.DataFrame:
    chunks = []
    for chunk_start, chunk_end in monthly_chunks(start, end):
        for attempt in range(1, retries + 1):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    df = statcast(start_dt=chunk_start, end_dt=chunk_end)
                chunks.append(df)
                break
            except Exception as e:
                if attempt < retries:
                    print(f"  Warning: {chunk_start}–{chunk_end} failed ({e}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"  Skipping {chunk_start}–{chunk_end} after {retries} failed attempts: {e}")
    return pd.concat(chunks, ignore_index=True)
