# trading_bot/src/state.py
import json
import time
from datetime import datetime
from threading import Lock, RLock

from . import config
from .config import logger, IS_GITHUB_ACTIONS
from .telegram_notifications import TelegramNotifier

# Initialize Telegram notifier
telegram_notifier = TelegramNotifier(
    bot_token=config.config.TELEGRAM_BOT_TOKEN, chat_id=config.config.TELEGRAM_CHAT_ID
)

# Initialize global state
portfolio = {"cash": config.config.PORTFOLIO_VALUE, "assets": {}}
portfolio_lock = RLock()  # Use RLock instead of Lock
rate_limit_lock = Lock()

low_volatility_assets = set()
negative_momentum_counts = {}
weight_used = 0
last_reset_time = time.time()
is_banned = False
ban_expiry_time = 0

def load_portfolio():
    """Loads the portfolio from the environment-specific file path or its backups."""
    global portfolio
    portfolio_path = config.config.PORTFOLIO_FILE
    try:
        with open(portfolio_path, "r") as f:
            portfolio_data = json.load(f)
            portfolio["cash"] = portfolio_data["cash"]
            portfolio["assets"] = {
                symbol: {
                    key: (
                        datetime.fromisoformat(value)
                        if key in ["purchase_time"]
                        else value
                    )
                    for key, value in asset.items()
                }
                for symbol, asset in portfolio_data["assets"].items()
            }
        logger.info(f"Loaded portfolio from {portfolio_path}")
    except Exception as e:
        logger.error(
            f"Error loading portfolio from {portfolio_path}: {e}", exc_info=True
        )
        telegram_notifier.notify_error(
            "Portfolio Load Failure", f"Error loading portfolio from {portfolio_path}: {e}"
        )
        # Try loading from latest backup
        import glob

        backup_files = sorted(
            glob.glob(f"{portfolio_path}.backup_*"), reverse=True
        )
        for backup_file in backup_files:
            try:
                with open(backup_file, "r") as f:
                    portfolio_data = json.load(f)
                    portfolio["cash"] = portfolio_data["cash"]
                    portfolio["assets"] = {
                        symbol: {
                            key: (
                                datetime.fromisoformat(value)
                                if key in ["purchase_time"]
                                else value
                            )
                            for key, value in asset.items()
                        }
                        for symbol, asset in portfolio_data["assets"].items()
                    }
                logger.info(f"Loaded portfolio from backup {backup_file}")
                break
            except Exception as backup_e:
                logger.error(
                    f"Error loading backup {backup_file}: {backup_e}", exc_info=True
                )
                telegram_notifier.notify_error(
                    "Portfolio Backup Load Failure", f"Error loading backup {backup_file}: {backup_e}"
                )
                continue
        else:
            logger.warning(
                "No valid portfolio file found. Initializing default portfolio."
            )
            portfolio = {"cash": config.config.PORTFOLIO_VALUE, "assets": {}}
            telegram_notifier.notify_error(
                "Portfolio Initialization",
                f"No valid portfolio file found. Initialized default portfolio with {config.config.PORTFOLIO_VALUE} EUR."
            )

def save_state():
    """Saves the bot state to the environment-specific file path."""
    state_path = "/tmp/state.json" if IS_GITHUB_ACTIONS else "state.json"
    try:
        state = {
            "low_volatility_assets": list(low_volatility_assets),
            "negative_momentum_counts": negative_momentum_counts,
            "weight_used": weight_used,
            "last_reset_time": last_reset_time,
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=4)
        logger.info(f"Saved state to {state_path}")
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}", exc_info=True)
        telegram_notifier.notify_error(
            "State Save Failure", f"Error saving state to {state_path}: {e}"
        )

def load_state():
    """Loads the bot state from the environment-specific file path."""
    global low_volatility_assets, negative_momentum_counts, weight_used, last_reset_time
    state_path = "/tmp/state.json" if IS_GITHUB_ACTIONS else "state.json"
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
            low_volatility_assets = set(state.get("low_volatility_assets", []))
            negative_momentum_counts = state.get("negative_momentum_counts", {})
            weight_used = state.get("weight_used", 0)
            last_reset_time = state.get("last_reset_time", time.time())
        logger.info(f"Loaded state from {state_path}")
    except Exception as e:
        logger.error(f"Error loading state from {state_path}: {e}", exc_info=True)
        telegram_notifier.notify_error(
            "State Load Failure", f"Error loading state from {state_path}: {e}"
        )

# Call load_portfolio() and load_state() at startup
load_portfolio()
load_state()