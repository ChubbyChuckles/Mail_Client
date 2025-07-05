# trading_bot/tests/test_utils.py
import logging
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.utils import calculate_dynamic_ema_period


# Mock config for IS_GITHUB_ACTIONS
class MockConfig:
    IS_GITHUB_ACTIONS = False


# Setup logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCalculateDynamicEMAPeriod(unittest.TestCase):
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
        self.assertEqual(result, 7)  # Base 5 + 2
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
        self.assertEqual(result, 8)  # Base 5 + 3
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
        self.assertEqual(
            result, 10
        )  # Base 5 + 2 + 3, capped at 10 (due to active_assets)
        self.mock_logger.debug.assert_called_once()
        self.mock_logger.error.assert_not_called()
        self.mock_send_alert.assert_not_called()

    def test_max_period_cap(self):
        """Test period cap at 12 when both conditions increase period significantly."""
        result = calculate_dynamic_ema_period(
            holding_minutes=300.0,
            time_stop_minutes=300,
            active_assets=10,
            asset_threshold=5,
        )
        self.assertEqual(
            result, 10
        )  # Base 5 + 2 (active_assets) + 3 (holding), capped at 10
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
        self.assertEqual(result, 5)  # No adjustments, base period
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
        # Mock logger.debug to raise an exception
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
