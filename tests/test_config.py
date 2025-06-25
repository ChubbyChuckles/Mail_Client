# MAIL_CLIENT_TEST/tests/test_config.py
import logging
import os
import sys
from unittest.mock import mock_open, patch

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (API_KEY, API_SECRET, ASSET_THRESHOLD,
                        CONCURRENT_REQUESTS, MAX_ACTIVE_ASSETS,
                        PORTFOLIO_VALUE, RATE_LIMIT_WEIGHT, TELEGRAM_BOT_TOKEN,
                        TELEGRAM_CHAT_ID, log_filename, logger)


@pytest.fixture(autouse=True)
def setup_env(monkeypatch, tmp_path):
    """Set up environment variables and temporary log directory."""
    env_vars = {
        "BITVAVO_API_KEY": "test_api_key",
        "BITVAVO_API_SECRET": "test_api_secret",
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "test_chat_id",
        "CONCURRENT_REQUESTS": "30",
        "RATE_LIMIT_WEIGHT": "1000",
        "PORTFOLIO_VALUE": "10000",
        "MAX_ACTIVE_ASSETS": "20",
        "RESULTS_FOLDER": str(tmp_path / "data"),
        "PARQUET_FILENAME": "test.parquet",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    yield
    # Cleanup
    monkeypatch.delenv("BITVAVO_API_KEY", raising=False)


def test_config_loading():
    """Test that environment variables are loaded correctly."""
    assert API_KEY == "test_api_key"
    assert API_SECRET == "test_api_secret"
    assert TELEGRAM_BOT_TOKEN == "test_token"
    assert TELEGRAM_CHAT_ID == "test_chat_id"
    assert CONCURRENT_REQUESTS == 30
    assert RATE_LIMIT_WEIGHT == 1000
    assert PORTFOLIO_VALUE == 10000
    assert MAX_ACTIVE_ASSETS == 20
    assert ASSET_THRESHOLD == int(20 * 0.6)


def test_logging_setup(tmp_path):
    """Test logging setup creates log file and handlers."""
    assert os.path.exists(log_filename)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 3  # Console, File, Evaluation
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
