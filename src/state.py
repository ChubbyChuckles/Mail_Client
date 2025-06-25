# trading_bot/src/state.py
import json
import time
from datetime import datetime
from threading import Lock, RLock

from .config import PORTFOLIO_FILE, PORTFOLIO_VALUE, logger

# Initialize global state
portfolio = {"cash": PORTFOLIO_VALUE, "assets": {}}
portfolio_lock = RLock()  # Use RLock instead of Lock
rate_limit_lock = Lock()

low_volatility_assets = set()
negative_momentum_counts = {}
weight_used = 0
last_reset_time = time.time()
is_banned = False
ban_expiry_time = 0


def load_portfolio():
    global portfolio
    try:
        with open(PORTFOLIO_FILE, "r") as f:
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
        logger.info(f"Loaded portfolio from {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(
            f"Error loading portfolio from {PORTFOLIO_FILE}: {e}", exc_info=True
        )
        # Try loading from latest backup
        import glob

        backup_files = sorted(glob.glob(f"{PORTFOLIO_FILE}.backup_*"), reverse=True)
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
                continue
        else:
            logger.warning(
                "No valid portfolio file found. Initializing default portfolio."
            )
            portfolio = {"cash": PORTFOLIO_VALUE, "assets": {}}


def save_state():
    try:
        state = {
            "low_volatility_assets": list(low_volatility_assets),
            "negative_momentum_counts": negative_momentum_counts,
            "weight_used": weight_used,
            "last_reset_time": last_reset_time,
        }
        with open("state.json", "w") as f:
            json.dump(state, f, indent=4)
        logger.info("Saved state to state.json")
    except Exception as e:
        logger.error(f"Error saving state: {e}", exc_info=True)


def load_state():
    global low_volatility_assets, negative_momentum_counts, weight_used, last_reset_time
    try:
        with open("state.json", "r") as f:
            state = json.load(f)
            low_volatility_assets = set(state.get("low_volatility_assets", []))
            negative_momentum_counts = state.get("negative_momentum_counts", {})
            weight_used = state.get("weight_used", 0)
            last_reset_time = state.get("last_reset_time", time.time())
        logger.info("Loaded state from state.json")
    except Exception as e:
        logger.error(f"Error loading state: {e}", exc_info=True)


# Call load_portfolio() at startup
load_portfolio()
load_state()
