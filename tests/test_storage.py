# MAIL_CLIENT_TEST/tests/test_storage.py
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
import os

from src.storage import save_to_local, save_portfolio
from src.state import portfolio, portfolio_lock

@pytest.fixture
def sample_df():
    """Create a sample DataFrame."""
    return pd.DataFrame({
        "timestamp": [pd.Timestamp("2021-06-30")],
        "symbol": ["BTC/EUR"],
        "close": [50000.0]
    })

def test_save_to_local_new_file(tmp_path, sample_df):
    """Test saving DataFrame to a new Parquet file."""
    output_path = tmp_path / "data/test.parquet"
    save_to_local(sample_df, str(output_path))
    assert output_path.exists()
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == 1
    assert saved_df["symbol"].iloc[0] == "BTC/EUR"

def test_save_to_local_append(tmp_path, sample_df):
    """Test appending to an existing Parquet file."""
    output_path = tmp_path / "data/test.parquet"
    save_to_local(sample_df, str(output_path))
    new_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2021-07-01")],
        "symbol": ["BTC/EUR"],
        "close": [51000.0]
    })
    save_to_local(new_df, str(output_path))
    saved_df = pd.read_parquet(output_path)
    assert len(saved_df) == 2

def test_save_portfolio(tmp_path):
    """Test saving portfolio to JSON."""
    portfolio_file = tmp_path / "portfolio.json"
    with portfolio_lock:
        portfolio["cash"] = 10000.0
        portfolio["assets"] = {"BTC/EUR": {"quantity": 0.1}}
    with patch("MAIL_CLIENT_TEST.src.config.PORTFOLIO_FILE", str(portfolio_file)):
        save_portfolio()
    assert portfolio_file.exists()