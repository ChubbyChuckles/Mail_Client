# tests/test_exchange.py
import pytest
import pandas as pd
import asyncio
from src.exchange import fetch_klines, initialize_exchange

@pytest.mark.asyncio
async def test_fetch_klines():
    exchange = await initialize_exchange()
    try:
        df = await fetch_klines("BTC/EUR", exchange)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'timestamp' in df.columns
        assert 'symbol' in df.columns
        assert df['symbol'].iloc[0] == "BTC/EUR"
    finally:
        await exchange.close()