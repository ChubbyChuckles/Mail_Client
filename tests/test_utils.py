# tests/test_utils.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.utils import calculate_ema

# Fixture to mock the logger
@pytest.fixture
def mock_logger(mocker):
    logger = mocker.patch("src.utils.logger")
    return logger

def test_calculate_ema_valid_list(mock_logger):
    """Test EMA calculation with a valid list of prices."""
    prices = [100, 101, 102, 103, 104]
    period = 3
    result = calculate_ema(prices, period)
    
    # Expected EMA calculation: (104 * 2/4 + 103 * 1/4 + 102 * 1/4) = 103.25
    # Using pandas ewm with span=3, adjust=False
    expected = pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
    assert isinstance(result, float)
    assert pytest.approx(result, expected, abs=1e-4)
    mock_logger.debug.assert_called_once_with(f"Calculated EMA for period {period}: {result:.4f}")
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_valid_series(mock_logger):
    """Test EMA calculation with a pandas Series."""
    prices = pd.Series([100, 101, 102, 103, 104])
    period = 3
    result = calculate_ema(prices, period)
    
    expected = prices.ewm(span=period, adjust=False).mean().iloc[-1]
    assert isinstance(result, float)
    assert pytest.approx(result, expected, abs=1e-4)
    mock_logger.debug.assert_called_once_with(f"Calculated EMA for period {period}: {result:.4f}")
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_insufficient_data(mock_logger):
    """Test EMA with fewer prices than the period."""
    prices = [100, 101]
    period = 3
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.warning.assert_called_once_with(
        f"Insufficient data for EMA calculation: {len(prices)} prices, need {period}"
    )
    mock_logger.debug.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_empty_prices(mock_logger):
    """Test EMA with an empty price list."""
    prices = []
    period = 3
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.warning.assert_called_once_with(
        f"Insufficient data for EMA calculation: 0 prices, need {period}"
    )
    mock_logger.debug.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_single_price(mock_logger):
    """Test EMA with a single price."""
    prices = [100]
    period = 3
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.warning.assert_called_once_with(
        f"Insufficient data for EMA calculation: 1 prices, need {period}"
    )
    mock_logger.debug.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_period_one(mock_logger):
    """Test EMA with period=1."""
    prices = [100, 101, 102]
    period = 1
    result = calculate_ema(prices, period)
    
    # With period=1, EMA is essentially the last price
    expected = pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
    assert isinstance(result, float)
    assert pytest.approx(result, expected, abs=1e-4)
    mock_logger.debug.assert_called_once_with(f"Calculated EMA for period {period}: {result:.4f}")
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()

def test_calculate_ema_invalid_prices(mock_logger):
    """Test EMA with non-numeric prices."""
    prices = [100, "invalid", 102]
    period = 3
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args[0][0].startswith("Error calculating EMA: ")
    mock_logger.warning.assert_not_called()
    mock_logger.debug.assert_not_called()

def test_calculate_ema_invalid_period(mock_logger):
    """Test EMA with invalid period (non-positive)."""
    prices = [100, 101, 102]
    period = 0
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args[0][0].startswith("Error calculating EMA: ")
    mock_logger.warning.assert_not_called()
    mock_logger.debug.assert_not_called()

def test_calculate_ema_none_input(mock_logger):
    """Test EMA with None as prices."""
    prices = None
    period = 3
    result = calculate_ema(prices, period)
    
    assert result is None
    mock_logger.error.assert_called_once()
    assert mock_logger.error.call_args[0][0].startswith("Error calculating EMA: ")
    mock_logger.warning.assert_not_called()
    mock_logger.debug.assert_not_called()