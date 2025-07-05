# trading_bot/src/send_portfolio.py
import json
import os
from datetime import datetime
import logging
import telegram  # Use python-telegram-bot==13.7 for synchronous API
from . import config
from .config import IS_GITHUB_ACTIONS

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if IS_GITHUB_ACTIONS:
    handler = logging.StreamHandler()
else:
    os.makedirs("trading_logs", exist_ok=True)
    handler = logging.FileHandler("trading_logs/send_portfolio.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def format_portfolio_message(portfolio_data):
    """Formats the portfolio data into a visually appealing Telegram message."""
    try:
        # Extract cash and assets
        cash = portfolio_data.get("cash", 0.0)
        assets = portfolio_data.get("assets", {})
        
        # Calculate total portfolio value
        total_asset_value = sum(
            asset["quantity"] * asset["current_price"]
            for asset in assets.values()
        )
        total_value = cash + total_asset_value

        # Header
        message = ["📊 *Portfolio Summary* 📊"]
        message.append(f"💶 *Cash*: €{cash:,.2f}")
        message.append(f"📈 *Total Asset Value*: €{total_asset_value:,.2f}")
        message.append(f"💰 *Total Portfolio Value*: €{total_value:,.2f}")
        message.append("")

        # Asset details
        message.append("📋 *Assets*:")
        if not assets:
            message.append("No assets held.")
        else:
            for symbol, asset in assets.items():
                # Calculate unrealized profit/loss
                purchase_price = asset["purchase_price"]
                current_price = asset["current_price"]
                quantity = asset["quantity"]
                unrealized_pl = (current_price - purchase_price) * quantity
                unrealized_pl_percent = ((current_price / purchase_price) - 1) * 100
                
                # Calculate holding time
                purchase_time = datetime.fromisoformat(asset["purchase_time"])
                holding_time = datetime.utcnow() - purchase_time
                holding_minutes = holding_time.total_seconds() / 60

                # Format asset details
                message.append(f"🔸 *{symbol}*")
                message.append(f"  Quantity: {quantity:,.2f}")
                message.append(f"  Current Price: €{current_price:,.4f}")
                message.append(f"  Purchase Price: €{purchase_price:,.4f}")
                message.append(f"  Unrealized P/L: €{unrealized_pl:,.2f} ({unrealized_pl_percent:+.2f}%)")
                message.append(f"  Holding Time: {holding_minutes:.1f} minutes")
                message.append(f"  Highest Price: €{asset['highest_price']:,.4f}")
                message.append(f"  Sell Price: €{asset['sell_price']:,.4f}")
                message.append(f"  Buy Fee: €{asset['buy_fee']:,.2f}")
                message.append(f"  Buy Slippage: {asset['buy_slippage']:.4%}")
                message.append("")

        # Footer with timestamp
        message.append(f"🕒 *Updated*: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return "\n".join(message)
    except Exception as e:
        logger.error(f"Error formatting portfolio message: {e}", exc_info=True)
        return f"⚠️ Error formatting portfolio data: {str(e)}"

def send_startup_message():
    """Sends a Telegram message signaling the bot has started."""
    bot_token = config.config.TELEGRAM_BOT_TOKEN
    chat_id = config.config.TELEGRAM_CHAT_ID

    if not bot_token or not chat_id:
        logger.error("Telegram bot token or chat ID not configured")
        return

    try:
        bot = telegram.Bot(token=bot_token)
        message = (
            "🚀 *Trading Bot Started* 🚀\n"
            "✅ *Status*: Initialized and running\n"
            f"🕒 *Started*: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"🌐 *Environment*: {'GitHub Actions' if IS_GITHUB_ACTIONS else 'Local'}"
        )
        bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=telegram.ParseMode.MARKDOWN
        )
        logger.info("Sent startup message via Telegram")
    except telegram.error.InvalidToken:
        logger.error("Invalid Telegram bot token")
    except telegram.error.TelegramError as e:
        logger.error(f"Telegram error sending startup message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error sending startup message: {e}", exc_info=True)

def send_shutdown_message():
    """Sends a Telegram message signaling the bot has shut down."""
    bot_token = config.config.TELEGRAM_BOT_TOKEN
    chat_id = config.config.TELEGRAM_CHAT_ID

    if not bot_token or not chat_id:
        logger.error("Telegram bot token or chat ID not configured")
        return

    try:
        bot = telegram.Bot(token=bot_token)
        message = (
            "🛑 *Trading Bot Shut Down* 🛑\n"
            "✅ *Status*: Gracefully stopped\n"
            f"🕒 *Stopped*: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"🌐 *Environment*: {'GitHub Actions' if IS_GITHUB_ACTIONS else 'Local'}"
        )
        bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=telegram.ParseMode.MARKDOWN
        )
        logger.info("Sent shutdown message via Telegram")
    except telegram.error.InvalidToken:
        logger.error("Invalid Telegram bot token")
    except telegram.error.TelegramError as e:
        logger.error(f"Telegram error sending shutdown message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error sending shutdown message: {e}", exc_info=True)

def send_portfolio():
    """Reads portfolio.json and sends the formatted data via Telegram synchronously."""
    portfolio_path = "/tmp/portfolio.json" if IS_GITHUB_ACTIONS else "portfolio.json"
    bot_token = config.config.TELEGRAM_BOT_TOKEN
    chat_id = config.config.TELEGRAM_CHAT_ID

    if not bot_token or not chat_id:
        logger.error("Telegram bot token or chat ID not configured")
        return

    try:
        bot = telegram.Bot(token=bot_token)
    except telegram.error.InvalidToken:
        logger.error("Invalid Telegram bot token")
        return

    try:
        # Read portfolio.json
        if not os.path.exists(portfolio_path):
            logger.error(f"Portfolio file not found: {portfolio_path}")
            bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ *Portfolio File Error*\nPortfolio file not found: {portfolio_path}",
                parse_mode=telegram.ParseMode.MARKDOWN
            )
            return

        with open(portfolio_path, "r") as f:
            portfolio_data = json.load(f)

        # Format and send message
        message = format_portfolio_message(portfolio_data)
        bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=telegram.ParseMode.MARKDOWN
        )
        logger.info(f"Sent portfolio update via Telegram from {portfolio_path}")
    except telegram.error.TelegramError as e:
        logger.error(f"Telegram error sending portfolio update: {e}", exc_info=True)
        bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ *Portfolio Send Failure*\nTelegram error: {str(e)}",
            parse_mode=telegram.ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error sending portfolio update: {e}", exc_info=True)
        try:
            bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ *Portfolio Send Failure*\nError: {str(e)}",
                parse_mode=telegram.ParseMode.MARKDOWN
            )
        except telegram.error.TelegramError as te:
            logger.error(f"Failed to send error notification: {te}", exc_info=True)

if __name__ == "__main__":
    send_portfolio()