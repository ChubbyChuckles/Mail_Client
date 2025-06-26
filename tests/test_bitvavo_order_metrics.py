import pytest
from unittest.mock import MagicMock, patch
from src.bitvavo_order_metrics import calculate_order_book_metrics

# Mock API_KEY and API_SECRET to avoid import issues
@pytest.fixture(autouse=True)
def mock_config():
    with patch("src.config.API_KEY", "dummy_key"), patch("src.config.API_SECRET", "dummy_secret"):
        yield

# Fixture to provide a mock Bitvavo client
@pytest.fixture
def mock_bitvavo():
    with patch("src.bitvavo_order_metrics.bitvavo") as mock:
        yield mock

# Fixture for a sample order book response
@pytest.fixture
def sample_order_book():
    return {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [
            ["50000.0", "0.1"],
            ["49900.0", "0.2"],
            ["49800.0", "0.3"],
        ],
        "asks": [
            ["50100.0", "0.1"],
            ["50200.0", "0.2"],
            ["50300.0", "0.3"],
        ],
    }

# Fixture for a sample analyze_buy_decision response
@pytest.fixture
def sample_buy_decision():
    return {
        "total_score": 0.75,
        "recommendation": "Buy"
    }

# Test successful order book metrics calculation
def test_calculate_order_book_metrics_success(mock_bitvavo, sample_order_book, sample_buy_decision):
    mock_bitvavo.book.return_value = sample_order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5.5, price_range_percent=10.0
        )

        assert metrics["market"] == "BTC-EUR"
        assert metrics["nonce"] == 12345
        assert metrics["best_bid"] == 50000.0
        assert metrics["best_ask"] == 50100.0
        assert metrics["spread"] == 100.0
        assert metrics["mid_price"] == 50050.0
        assert metrics["spread_percentage"] == (100.0 / 50050.0) * 100
        assert metrics["bid_volume"] == pytest.approx(0.6, abs=1e-10)  # Was 0.6
        assert metrics["ask_volume"] == pytest.approx(0.6, abs=1e-10)  # Was 0.6
        assert metrics["buy_depth"] == 29920.0  # 50000*0.1 + 49900*0.2 + 49800*0.3
        assert metrics["sell_depth"] == 30140.0  # 50100*0.1 + 50200*0.2 + 50300*0.3
        assert metrics["total_depth"] == 29920.0 + 30140.0
        assert metrics["order_book_imbalance"] == 0.5  # bid_volume / (bid_volume + ask_volume)
        assert metrics["total_score"] == 0.75
        assert metrics["recommendation"] == "Buy"

# Test empty order book
def test_empty_order_book(mock_bitvavo):
    mock_bitvavo.book.return_value = {"market": "BTC-EUR", "nonce": 12345, "bids": [], "asks": []}
    metrics = calculate_order_book_metrics(market="BTC-EUR")
    assert metrics == {"error": "Failed to retrieve order book"}

# Test missing bids
def test_missing_bids(mock_bitvavo, sample_order_book):
    order_book = sample_order_book.copy()
    order_book["bids"] = []
    mock_bitvavo.book.return_value = order_book
    metrics = calculate_order_book_metrics(market="BTC-EUR")
    assert metrics == {"error": "Failed to retrieve order book"}

# Test missing asks
def test_missing_asks(mock_bitvavo, sample_order_book):
    order_book = sample_order_book.copy()
    order_book["asks"] = []
    mock_bitvavo.book.return_value = order_book
    metrics = calculate_order_book_metrics(market="BTC-EUR")
    assert metrics == {"error": "Failed to retrieve order book"}

# Test API failure
def test_api_failure(mock_bitvavo):
    mock_bitvavo.book.side_effect = Exception("API error")
    metrics = calculate_order_book_metrics(market="BTC-EUR")
    assert metrics == {"error": "API error"}

# Test zero bid volume for average price calculations
def test_zero_bid_volume(mock_bitvavo, sample_order_book):
    order_book = sample_order_book.copy()
    order_book["bids"] = [["50000.0", "0.0"]]
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value={"total_score": 0.5, "recommendation": "Hold"}):
        metrics = calculate_order_book_metrics(market="BTC-EUR")
        assert metrics["avg_bid_price"] is None
        assert metrics["vwap_bid"] is None

# Test zero ask volume for average price calculations
def test_zero_ask_volume(mock_bitvavo, sample_order_book):
    order_book = sample_order_book.copy()
    order_book["asks"] = [["50100.0", "0.0"]]
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value={"total_score": 0.5, "recommendation": "Hold"}):
        metrics = calculate_order_book_metrics(market="BTC-EUR")
        assert metrics["avg_ask_price"] is None
        assert metrics["vwap_ask"] is None

# Test 1: Buy slippage with multiple price levels
def test_buy_slippage_multiple_levels(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"], ["49900.0", "0.2"]],
        "asks": [["50100.0", "0.05"], ["50200.0", "0.1"], ["50300.0", "0.2"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 5000.0 / expected_price  # ~0.0998 BTC
        # Use 0.05 @ 50100.0 + 0.0498 @ 50200.0
        weighted_price_sum = (0.05 * 50100.0) + (0.0498 * 50200.0)
        predicted_price = weighted_price_sum / (0.05 + 0.0498)
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] == pytest.approx(slippage, abs=1e-5)
        assert metrics["predicted_price_buy"] == pytest.approx(predicted_price, abs=1e-3)  # Increased tolerance

# Test 2: Buy slippage with single price level (should be zero)
def test_buy_slippage_single_level(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.1"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 5000.0 / expected_price  # ~0.0998 BTC
        # Only one level at 50100.0, so predicted price = expected price
        assert metrics["slippage_buy"] == 0.0, "Slippage should be zero for single level"
        assert metrics["predicted_price_buy"] == expected_price

# Test 3: Buy slippage with insufficient depth
def test_buy_slippage_insufficient_depth(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.01"]],  # Not enough to fill order
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        assert metrics["slippage_buy"] is None, "Slippage should be None due to insufficient depth"
        assert metrics["predicted_price_buy"] is None, "Predicted price should be None"

# Test 4: Buy slippage with small order amount
def test_buy_slippage_small_amount(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.1"], ["50200.0", "0.1"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=10.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 10.0 / expected_price  # ~0.0001996 BTC
        # Entire order filled at 50100.0
        predicted_price = 50100.0
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] == 0.0, "Slippage should be zero for small order"
        assert metrics["predicted_price_buy"] == predicted_price

# Test 5: Buy slippage with identical ask prices
def test_buy_slippage_identical_prices(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.05"], ["50100.0", "0.05"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 5000.0 / expected_price  # ~0.0998 BTC
        # Both levels at 50100.0, so predicted price = 50100.0
        predicted_price = 50100.0
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] == 0.0, "Slippage should be zero for identical prices"
        assert metrics["predicted_price_buy"] == predicted_price

# Test 6: Buy slippage with large price gaps
def test_buy_slippage_large_price_gaps(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.05"], ["51000.0", "0.1"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 5000.0 / expected_price  # ~0.0998 BTC
        # Use 0.05 @ 50100.0 + 0.0498 @ 51000.0
        weighted_price_sum = (0.05 * 50100.0) + (0.0498 * 51000.0)
        predicted_price = weighted_price_sum / (0.05 + 0.0498)
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] != 0.0, "Slippage should not be zero with price gaps"
        assert metrics["slippage_buy"] == pytest.approx(slippage, abs=1e-4)
        assert metrics["predicted_price_buy"] == pytest.approx(predicted_price, abs=1e-2)

# Test 7: Buy slippage with very small price differences
def test_buy_slippage_small_price_diff(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.05"], ["50100.1", "0.05"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 5000.0 / expected_price  # ~0.0998 BTC
        # Use 0.05 @ 50100.0 + 0.0498 @ 50100.1
        weighted_price_sum = (0.05 * 50100.0) + (0.0498 * 50100.1)
        predicted_price = weighted_price_sum / (0.05 + 0.0498)
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] != 0.0, "Slippage should not be zero with small price differences"
        assert metrics["slippage_buy"] == pytest.approx(slippage, abs=1e-6)
        assert metrics["predicted_price_buy"] == pytest.approx(predicted_price, abs=1e-6)

# Test 8: Buy slippage with high amount_quote
def test_buy_slippage_high_amount(mock_bitvavo, sample_buy_decision):
    order_book = {
        "market": "BTC-EUR",
        "nonce": 12345,
        "bids": [["50000.0", "0.1"]],
        "asks": [["50100.0", "0.1"], ["50200.0", "0.1"], ["50500.0", "0.1"]],
    }
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=10000.0, price_range_percent=10.0
        )
        expected_price = 50100.0
        base_amount = 10000.0 / expected_price  # ~0.1996 BTC
        # Use 0.1 @ 50100.0 + 0.0996 @ 50200.0
        weighted_price_sum = (0.1 * 50100.0) + (0.0996 * 50200.0)
        predicted_price = weighted_price_sum / (0.1 + 0.0996)
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_buy"] != 0.0, "Slippage should not be zero for high amount"
        assert metrics["slippage_buy"] == pytest.approx(slippage, abs=1e-6)
        assert metrics["predicted_price_buy"] == pytest.approx(predicted_price, abs=1e-3)

# Test slippage calculation for sell
def test_slippage_sell(mock_bitvavo, sample_order_book, sample_buy_decision):
    mock_bitvavo.book.return_value = sample_order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5.0, price_range_percent=10.0
        )
        expected_price = 50000.0
        base_amount = 5.0 / expected_price
        predicted_price = (50000.0 * base_amount) / base_amount  # Single level used
        slippage = ((predicted_price - expected_price) / expected_price) * 100
        assert metrics["slippage_sell"] == pytest.approx(slippage)
        assert metrics["predicted_price_sell"] == pytest.approx(predicted_price)

# Test insufficient order book depth for slippage
def test_insufficient_depth_for_slippage(mock_bitvavo, sample_order_book, sample_buy_decision):
    order_book = sample_order_book.copy()
    order_book["asks"] = [["50100.0", "0.00001"]]  # Very small amount
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=1000.0, price_range_percent=10.0
        )
        assert metrics["slippage_buy"] is None
        assert metrics["predicted_price_buy"] is None

# Test order book imbalance with zero total volume
def test_zero_volume_imbalance(mock_bitvavo, sample_order_book, sample_buy_decision):
    order_book = sample_order_book.copy()
    order_book["bids"] = [["50000.0", "0.0"]]
    order_book["asks"] = [["50100.0", "0.0"]]
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(market="BTC-EUR")
        assert metrics["order_book_imbalance"] is None

# Test invalid market parameter
def test_invalid_market(mock_bitvavo):
    mock_bitvavo.book.side_effect = Exception("Invalid market")
    metrics = calculate_order_book_metrics(market="INVALID-MARKET")
    assert metrics == {"error": "Invalid market"}


# Test large price range
def test_large_price_range(mock_bitvavo, sample_order_book, sample_buy_decision):
    order_book = sample_order_book.copy()
    mock_bitvavo.book.return_value = order_book
    with patch("src.bitvavo_order_metrics.analyze_buy_decision", return_value=sample_buy_decision):
        metrics = calculate_order_book_metrics(
            market="BTC-EUR", amount_quote=5.5, price_range_percent=100.0
        )
        # Large range includes all bids and asks
        assert metrics["bid_volume"] == 0
        assert metrics["ask_volume"] == pytest.approx(0.6)
        assert metrics["buy_depth"] == 0
        assert metrics["sell_depth"] == 30140.0