import logging
import os
from datetime import datetime, timedelta
from unittest.mock import mock_open, patch

import pytest
from dotenv import load_dotenv

from src.config import (API_KEY, ASSET_THRESHOLD, MAX_ACTIVE_ASSETS,
                        PRICE_RANGE_PERCENT, TELEGRAM_BOT_TOKEN,
                        TELEGRAM_CHAT_ID, EvaluationLogHandler, logger,
                        parse_float_env)


# Fixture to reset logging handlers before each test
@pytest.fixture(autouse=True)
def reset_logging():
    logger.handlers = []
    yield
    logger.handlers = []


# Test logging setup
def test_evaluation_log_handler(tmp_path):
    """Test that EvaluationLogHandler writes INFO logs with 'Evaluation Decision' to file."""
    # Ensure trading_logs directory exists in tmp_path
    log_dir = tmp_path / "trading_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "test.log"

    # Reset logger handlers to avoid conflicts with config.py
    logger.handlers = []

    # Mock file writing for EvaluationLogHandler
    mock_file = mock_open()
    with patch("src.config.log_filename", str(log_file)), patch(
        "builtins.open", mock_file
    ):
        # Configure logger with EvaluationLogHandler
        handler = EvaluationLogHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Test INFO log with 'Evaluation Decision'
        logger.info("Evaluation Decision: Test decision")
        mock_file().write.assert_called_once()
        written_content = mock_file().write.call_args[0][0]
        assert "Evaluation Decision: Test decision" in written_content
        assert "[INFO]" in written_content

        # Test regular INFO log (should not write)
        mock_file.reset_mock()
        logger.info("Regular log message")
        mock_file().write.assert_not_called()

        # Test another evaluation log
        mock_file.reset_mock()
        logger.info("Evaluation Decision: Another decision")
        mock_file().write.assert_called_once()
        written_content = mock_file().write.call_args[0][0]
        assert "Evaluation Decision: Another decision" in written_content
        assert "[INFO]" in written_content


# Test parse_float_env
@patch.dict(os.environ, {"TEST_VAR": "1.5 # comment"})
def test_parse_float_env_valid():
    """Test parse_float_env with valid input."""
    assert parse_float_env("TEST_VAR", 0.0) == 1.5


@patch.dict(os.environ, {"TEST_VAR": "invalid"})
def test_parse_float_env_invalid():
    """Test parse_float_env with invalid input, falls back to default."""
    with patch.object(logger, "error") as mock_error:
        result = parse_float_env("TEST_VAR", 0.0)
        assert result == 0.0
        mock_error.assert_called_with(
            "Invalid value for TEST_VAR: invalid. Using default: 0.0"
        )


@patch.dict(os.environ, {})
def test_parse_float_env_missing():
    """Test parse_float_env with missing environment variable."""
    assert parse_float_env("MISSING_VAR", 2.0) == 2.0
