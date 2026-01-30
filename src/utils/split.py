from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def time_split_masks(
        df: pd.DataFrame,
        time_col: str,
        train_frac: float,
        valid_frac: float,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(df[time_col].to_numpy())
    n =len(df)
    n_train = int(n* train_frac)
    n_valid = int(n * valid_frac)

    train_idx = order[:n_train]
    valid_idx = order[n_train:n_train+n_valid]
    test_idx =  order[n_train + n_valid:]

    train_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True
    return train_mask, valid_mask, test_mask
