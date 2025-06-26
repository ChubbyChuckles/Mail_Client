# tests/test_data_processor.py
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging
from src.data_processor import verify_and_analyze_data, colorize_value

# Sample configuration values for testing
MIN_VOLUME_EUR = 10000
PRICE_INCREASE_THRESHOLD = 5.0
MAX_SLIPPAGE_BUY = 0.5
MAX_SLIPPAGE_SELL = -0.5
ALLOCATION_PER_TRADE = 0.1
PRICE_RANGE_PERCENT = 2.0
MIN_TOTAL_SCORE = 0.7

@pytest.fixture
def sample_ohlcv_data():
    """Fixture for sample OHLCV data."""
    current_time = datetime.utcnow()
    five_min_ago = current_time - timedelta(minutes=5)
    ten_min_ago = current_time - timedelta(minutes=10)
    data = {
        "timestamp": [
            five_min_ago, five_min_ago, ten_min_ago,
            five_min_ago, five_min_ago, ten_min_ago
        ],
        "symbol": ["BTC/EUR", "ETH/EUR", "BTC/EUR", "ADA/EUR", "XRP/EUR", "ADA/EUR"],
        "open": [50000.0, 3000.0, 49000.0, 1.0, 0.5, 0.95],
        "close": [52500.0, 3150.0, 49500.0, 1.05, 0.52, 0.98],
        "volume": [10.0, 1000.0, 15.0, 50000.0, 20000.0, 60000.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def price_monitor_manager():
    """Fixture for a mock price monitor manager."""
    return MagicMock()

@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger."""
    with patch("src.data_processor.logger") as mock:
        yield mock

@pytest.fixture
def mock_check_rate_limit():
    """Fixture for mocking check_rate_limit."""
    with patch("src.data_processor.check_rate_limit") as mock:
        yield mock

@pytest.fixture
def mock_calculate_order_book_metrics():
    """Fixture for mocking calculate_order_book_metrics."""
    with patch("src.data_processor.calculate_order_book_metrics") as mock:
        mock.side_effect = lambda market, **kwargs: {
            "market": market,
            "slippage_buy": 0.3,
            "slippage_sell": -0.2,
            "total_score": 0.8,
            "recommendation": "Strong Buy"
        }
        yield mock

@pytest.fixture
def mock_portfolio():
    """Fixture for mock portfolio."""
    return {"cash": 10000, "assets": {"BTC/EUR": 0.1, "ADA/EUR": 1000}}

@pytest.fixture
def mock_portfolio_lock():
    """Fixture for mock portfolio lock."""
    return MagicMock()

@pytest.fixture
def mock_low_volatility_assets():
    """Fixture for mock low_volatility_assets."""
    return set(["ADA/EUR"])

def test_empty_dataframe(mock_logger):
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    price_monitor_manager = MagicMock()
    result = verify_and_analyze_data(df, price_monitor_manager)
    
    assert result[0] == []
    assert result[1].empty
    assert result[2] == []
    mock_logger.warning.assert_called_with("Input DataFrame is empty.")

def test_old_data(mock_logger):
    """Test handling of data older than 10 minutes."""
    old_time = datetime.utcnow() - timedelta(minutes=15)
    df = pd.DataFrame({
        "timestamp": [old_time],
        "symbol": ["BTC/EUR"],
        "open": [50000.0],
        "close": [51000.0],
        "volume": [10.0]
    })
    price_monitor_manager = MagicMock()
    result = verify_and_analyze_data(df, price_monitor_manager)
    
    assert result[0] == []
    assert result[1].empty
    assert result[2] == []
    mock_logger.warning.assert_called_with("Data contains no candles from within the last 10 minutes.")

def test_no_recent_data(mock_logger):
    """Test handling of data with no candles within the last 5 minutes."""
    ten_min_ago = datetime.utcnow() - timedelta(minutes=10)
    df = pd.DataFrame({
        "timestamp": [ten_min_ago],
        "symbol": ["BTC/EUR"],
        "open": [50000.0],
        "close": [51000.0],
        "volume": [10.0]
    })
    price_monitor_manager = MagicMock()
    result = verify_and_analyze_data(df, price_monitor_manager)
    
    assert result[0] == []
    assert result[1].empty
    assert result[2] == []
    mock_logger.warning.assert_called_with("No recent data within the last 5 minutes.")

def test_above_threshold_assets(sample_ohlcv_data, price_monitor_manager, mock_logger, mock_check_rate_limit, mock_calculate_order_book_metrics):
    """Test assets meeting price increase and volume thresholds."""
    result = verify_and_analyze_data(sample_ohlcv_data, price_monitor_manager)
    
    above_threshold_data, percent_changes, order_book_metrics_list = result
    
    # Check above threshold assets
    assert len(above_threshold_data) >= 2, f"Expected at least 2 assets, got {len(above_threshold_data)}: {above_threshold_data}"
    assert any(d["symbol"] == "BTC/EUR" for d in above_threshold_data)
    assert any(d["symbol"] == "ETH/EUR" for d in above_threshold_data)
    
    # Check percent changes DataFrame
    assert not percent_changes.empty
    assert "percent_change" in percent_changes.columns
    assert "volume_eur" in percent_changes.columns
    
    # Check order book metrics
    assert len(order_book_metrics_list) >= 2
    assert all("bought" in metrics and metrics["bought"] is False for metrics in order_book_metrics_list)

def test_below_threshold_logging(sample_ohlcv_data, price_monitor_manager, mock_logger, mock_check_rate_limit, mock_calculate_order_book_metrics):
    """Test logging for assets below threshold."""
    verify_and_analyze_data(sample_ohlcv_data, price_monitor_manager)
    
    assert any("XRP/EUR" in call[0][0] for call in mock_logger.info.call_args_list), "Expected XRP/EUR in below-threshold logging"

def test_low_volatility_assets_resume_monitoring(sample_ohlcv_data, price_monitor_manager, mock_portfolio, mock_portfolio_lock, mock_low_volatility_assets, mock_logger, mock_check_rate_limit, mock_calculate_order_book_metrics):
    """Test resuming monitoring for low volatility assets with significant price change."""
    with patch("src.data_processor.low_volatility_assets", mock_low_volatility_assets):
        verify_and_analyze_data(sample_ohlcv_data, price_monitor_manager)
    
    assert "ADA/EUR" not in mock_low_volatility_assets
    price_monitor_manager.start.assert_called()


def test_invalid_data_handling():
    """Test handling of invalid data (e.g., zero open price)."""
    df = pd.DataFrame({
        "timestamp": [datetime.utcnow()],
        "symbol": ["BTC/EUR"],
        "open": [0.0],
        "close": [51000.0],
        "volume": [10.0]
    })
    price_monitor_manager = MagicMock()
    _, percent_changes, _ = verify_and_analyze_data(df, price_monitor_manager)
    
    assert percent_changes.empty

def test_no_above_threshold_assets(mock_logger, price_monitor_manager, mock_check_rate_limit, mock_calculate_order_book_metrics):
    """Test when no assets meet the threshold."""
    df = pd.DataFrame({
        "timestamp": [datetime.utcnow()],
        "symbol": ["BTC/EUR"],
        "open": [50000.0],
        "close": [50100.0],  # 0.2% increase
        "volume": [5.0]
    })
    result = verify_and_analyze_data(df, price_monitor_manager)
    
    assert result[0] == []
    assert any("No coins with price increase >= 5.0% and volume >= €10000" in call[0][0] for call in mock_logger.info.call_args_list)

def test_no_below_threshold_assets(mock_logger, price_monitor_manager, mock_check_rate_limit, mock_calculate_order_book_metrics):
    """Test when no assets are below threshold."""
    df = pd.DataFrame({
        "timestamp": [datetime.utcnow()],
        "symbol": ["BTC/EUR"],
        "open": [50000.0],
        "close": [52500.0],  # 5% increase
        "volume": [100.0]
    })
    result = verify_and_analyze_data(df, price_monitor_manager)
    
    assert not result[1].empty
    assert any("No coins with price increase < 5.0%" in call[0][0] for call in mock_logger.info.call_args_list)