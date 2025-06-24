# tests/test_utils.py
import pytest
import pandas as pd
from src.utils import calculate_ema

def test_calculate_ema():
    data = pd.Series([100, 101, 102, 103, 104])
    ema = calculate_ema(data, 3)
    assert isinstance(ema, pd.Series)
    assert len(ema) == len(data)
    assert all(ema.notna())