from datetime import datetime

MIN_N_EFF_PRIOR = 20      # for population param estimation
MIN_N_EFF_POSTERIOR = 5   # for posterior inference
DEFAULT_LAMBDA = 0.0116
CURRENT_YEAR = datetime.now().year
LOOKBACK_YEARS = [2025, CURRENT_YEAR]  # 2 years only