# trading_bot/tests/test_utils.py
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.utils import calculate_dynamic_ema_period, calculate_rsi


# Mock config for IS_GITHUB_ACTIONS
class MockConfig:
    IS_GITHUB_ACTIONS = False


# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up mocks for logger and send_alert before each test."""
        self.logger_patcher = patch("src.utils.logger", MagicMock())
        self.send_alert_patcher = patch("src.utils.send_alert", MagicMock())
        self.mock_logger = self.logger_patcher.start()
        self.mock_send_alert = self.send_alert_patcher.start()
        self.config_patcher = patch("src.utils.config", MockConfig())
        self.config_patcher.start()

    def tearDown(self):
        """Stop all patchers after each test."""
        self.logger_patcher.stop()
        self.send_alert_patcher.stop()
        self.config_patcher.stop()

    # Tests for calculate_dynamic_ema_period (from previous response)
    def test_normal_case_no_adjustments(self):
        """Test with holding_minutes < time_stop_minutes * 0.5 and active_assets < asset_threshold."""
        result = calculate_dynamic_ema_period(
            holding_minutes=100.0,
            time_stop_minutes=300,
            active_assets=2,
            asset_threshold=5,
        )
        self.assertEqual(result, 5)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_increase_by_active_assets(self):
        """Test when active_assets >= asset_threshold."""
        result = calculate_dynamic_ema_period(
            holding_minutes=100.0,
            time_stop_minutes=300,
            active_assets=5,
            asset_threshold=5,
        )
        self.assertEqual(result, 7)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_increase_by_holding_time(self):
        """Test when holding_minutes >= time_stop_minutes * 0.5."""
        result = calculate_dynamic_ema_period(
            holding_minutes=150.0,
            time_stop_minutes=300,
            active_assets=2,
            asset_threshold=5,
        )
        self.assertEqual(result, 8)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_both_adjustments(self):
        """Test when both conditions are met."""
        result = calculate_dynamic_ema_period(
            holding_minutes=150.0,
            time_stop_minutes=300,
            active_assets=5,
            asset_threshold=5,
        )
        self.assertEqual(result, 10)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_max_period_cap(self):
        """Test period cap at 10."""
        result = calculate_dynamic_ema_period(
            holding_minutes=300.0,
            time_stop_minutes=300,
            active_assets=10,
            asset_threshold=5,
        )
        self.assertEqual(result, 10)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_minimum_period(self):
        """Test period is at least 2."""
        result = calculate_dynamic_ema_period(
            holding_minutes=0.0,
            time_stop_minutes=300,
            active_assets=0,
            asset_threshold=5,
        )
        self.assertEqual(result, 5)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_non_numeric_input(self):
        """Test non-numeric input raises TypeError and returns default."""
        result = calculate_dynamic_ema_period(
            holding_minutes="invalid",
            time_stop_minutes=300,
            active_assets=2,
            asset_threshold=5,
        )
        self.assertEqual(result, 5)
        self.mock_logger.error.assert_called_once_with(
            "Error calculating dynamic EMA period: All inputs must be numeric",
            exc_info=True,
        )
        self.mock_send_alert.assert_not_called()

    def test_negative_input(self):
        """Test negative input raises ValueError and returns default."""
        result = calculate_dynamic_ema_period(
            holding_minutes=-100.0,
            time_stop_minutes=300,
            active_assets=2,
            asset_threshold=5,
        )
        self.assertEqual(result, 5)
        self.mock_logger.error.assert_called_once_with(
            "Error calculating dynamic EMA period: Inputs cannot be negative",
            exc_info=True,
        )
        self.mock_send_alert.assert_not_called()

    def test_unexpected_exception(self):
        """Test unexpected exception handling with send_alert."""
        self.mock_logger.debug.side_effect = RuntimeError("Unexpected error")
        result = calculate_dynamic_ema_period(
            holding_minutes=100.0,
            time_stop_minutes=300,
            active_assets=2,
            asset_threshold=5,
        )
        self.assertEqual(result, 5)
        self.mock_logger.error.assert_called_once_with(
            "Unexpected error calculating dynamic EMA period: Unexpected error",
            exc_info=True,
        )
        self.mock_send_alert.assert_called_once_with(
            "Dynamic EMA Period Failure",
            "Unexpected error calculating dynamic EMA period: Unexpected error",
        )

    # Tests for calculate_rsi
    def test_rsi_valid_input(self):
        """Test RSI calculation with valid input and default period."""
        closes = [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
        ]
        result = calculate_rsi(closes, period=14)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        self.mock_logger.error.assert_not_called()

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data returns None."""
        closes = [100, 101, 102]  # Less than period=14
        result = calculate_rsi(closes, period=14)
        self.assertIsNone(result)
        self.mock_logger.error.assert_not_called()

    def test_rsi_zero_losses(self):
        """Test RSI when all changes are gains (avg_loss = 0)."""
        closes = [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
        ]
        result = calculate_rsi(closes, period=14)
        self.assertEqual(result, 100)  # All gains -> RSI = 100
        self.mock_logger.error.assert_not_called()

    def test_rsi_zero_gains(self):
        """Test RSI when all changes are losses (avg_gain = 0)."""
        closes = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86]
        result = calculate_rsi(closes, period=14)
        self.assertEqual(result, 0)  # All losses -> RSI = 0
        self.mock_logger.error.assert_not_called()

    def test_rsi_custom_period(self):
        """Test RSI with a custom period."""
        closes = [100, 101, 100, 102, 103, 104, 105, 106]
        result = calculate_rsi(closes, period=7)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        self.mock_logger.error.assert_not_called()

    def test_rsi_empty_input(self):
        """Test RSI with empty input returns None."""
        closes = []
        result = calculate_rsi(closes, period=14)
        self.assertIsNone(result)
        self.mock_logger.error.assert_not_called()

    def test_rsi_numpy_array_input(self):
        """Test RSI with NumPy array input."""
        closes = np.array(
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        )
        result = calculate_rsi(closes, period=14)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 100)
        self.mock_logger.error.assert_not_called()
