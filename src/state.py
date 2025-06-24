# trading_bot/src/state.py
from threading import Lock
import json
from datetime import datetime
import time
from .config import PORTFOLIO_VALUE

# Initialize global state
portfolio = {'cash': PORTFOLIO_VALUE, 'assets': {}}
portfolio_lock = Lock()
rate_limit_lock = Lock()

low_volatility_assets = set()
negative_momentum_counts = {}
last_sheets_write = 0
last_sheets_load = 0
weight_used = 0
last_reset_time = time.time()