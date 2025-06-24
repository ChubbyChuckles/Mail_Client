# trading_bot/src/notifications.py
import asyncio

import telegram
from telegram.error import TelegramError

from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, logger

# Initialize Telegram bot
bot = None
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        logger.info("Telegram bot initialized successfully.")
    except TelegramError as e:
        logger.error(f"Failed to initialize Telegram bot: {e}")
        bot = None
else:
    logger.warning(
        "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing. Telegram notifications disabled."
    )

# Initialize a single event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def send_telegram_message(message: str):
    """
    Sends a message to the configured Telegram chat.

    Args:
        message (str): The message to send.
    """
    if bot is None:
        logger.debug("Telegram bot not initialized. Skipping message.")
        return
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.debug(f"Sent Telegram message: {message}")
    except TelegramError as e:
        logger.error(f"Failed to send Telegram message: {e}")


async def send_trade_notification(trade: dict, reason: str):
    """
    Sends a formatted trade notification to the configured Telegram chat.

    Args:
        trade (dict): Dictionary containing trade details.
        reason (str): Reason for the trade action.
    """
    if bot is None:
        logger.debug("Telegram bot not initialized. Skipping trade notification.")
        return
    try:
        message = (
            f"Trade Executed: {trade['Symbol']}\n"
            f"Action: Sell\n"
            f"Reason: {reason}\n"
            f"Buy Price: €{trade['Buy Price']}\n"
            f"Buy Time: {trade['Buy Time']}\n"
            f"Sell Price: €{trade['Sell Price']}\n"
            f"Sell Time: {trade['Sell Time']}\n"
            f"Quantity: {trade['Sell Quantity']}\n"
            f"Profit/Loss: €{trade['Profit/Loss']}\n"
            f"Buy Fee: €{trade['Buy Fee']}\n"
            f"Sell Fee: €{trade['Sell Fee']}"
        )
        await send_telegram_message(message)
    except Exception as e:
        logger.error(f"Failed to send trade notification: {e}")


def run_async(coro):
    """
    Runs an async coroutine in the existing event loop.

    Args:
        coro: The coroutine to run.
    """
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error running async coroutine: {e}")
        return None


def shutdown_loop():
    """
    Safely shuts down the event loop.
    """
    try:
        tasks = [
            t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task(loop)
        ]
        for task in tasks:
            task.cancel()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()
        logger.debug("Event loop shut down successfully.")
    except Exception as e:
        logger.error(f"Error shutting down event loop: {e}")
