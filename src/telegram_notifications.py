# trading_bot/telegram_notifications.py
import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime
from queue import Queue
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.error import RetryAfter, TelegramError
from telegram.ext import Application, CallbackQueryHandler
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from . import config
from .config import logger


class TelegramNotifier:
    """
    A class to send asynchronous Telegram notifications for trading bot activities.
    Includes inline keyboards, charts, pinned messages, performance alerts, and daily reports.
    Ensures non-blocking notifications with queuing and rate limiting.
    """

    def __init__(self, bot_token: str, chat_id: str):
        """
        Initializes the TelegramNotifier with bot token and chat ID.

        Args:
            bot_token (str): Telegram Bot API token.
            chat_id (str): Telegram chat ID to send messages to.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        self.queue = Queue()
        self.running = False
        self.rate_limit_delay = (
            3  # Seconds to wait between messages to avoid rate limits
        )
        self.logger = logger  # Reuse bot's logger
        self.pinned_message_id = None  # Store pinned message ID
        self.application = Application.builder().token(bot_token).build()
        self._setup_handlers()

    def _setup_handlers(self):
        """Sets up callback handlers for inline keyboards."""
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))

    async def start(self):
        """
        Starts the notification processing loop and callback polling in the background.
        """
        if not self.running:
            self.running = True
            asyncio.create_task(self._process_queue())
            self.application.run_polling(
                drop_pending_updates=True, stop_signals=[]
            )  # Non-blocking
            self.logger.info("TelegramNotifier started with queue and callback polling")

    async def stop(self):
        """
        Stops the notification processing loop and callback polling.
        """
        self.running = False
        await self.application.stop()
        self.logger.info("TelegramNotifier stopped")

    async def _process_queue(self):
        """
        Processes the notification queue asynchronously, respecting rate limits.
        """
        while self.running:
            if not self.queue.empty():
                item = self.queue.get()
                if len(item) == 2:  # Text message
                    message, is_html = item
                    try:
                        await self._send_message_with_retry(
                            message, parse_mode="HTML" if is_html else None
                        )
                        self.logger.debug(f"Sent Telegram message: {message[:50]}...")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to send Telegram message: {e}", exc_info=True
                        )
                elif len(item) == 3:  # Photo (chart)
                    photo, caption, is_html = item
                    try:
                        await self._send_photo_with_retry(
                            photo, caption, parse_mode="HTML" if is_html else None
                        )
                        self.logger.debug(f"Sent Telegram photo: {caption[:50]}...")
                    except Exception as e:
                        self.logger.error(
                            f"Failed to send Telegram photo: {e}", exc_info=True
                        )
                await asyncio.sleep(self.rate_limit_delay)
            else:
                await asyncio.sleep(1)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TelegramError, RetryAfter)),
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying Telegram message send after attempt {retry_state.attempt_number}"
        ),
    )
    async def _send_message_with_retry(
        self, message: str, parse_mode: Optional[str] = None
    ):
        """
        Sends a message with inline keyboard and retry logic.

        Args:
            message (str): The message to send.
            parse_mode (str, optional): Parse mode for the message (e.g., 'HTML').
        """
        keyboard = [
            [
                InlineKeyboardButton("Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("Trades", callback_data="trades"),
                InlineKeyboardButton("Stats", callback_data="stats"),
            ],
            [InlineKeyboardButton("Logs", callback_data="logs")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except RetryAfter as e:
            self.logger.warning(
                f"Telegram rate limit hit, retrying after {e.retry_after} seconds"
            )
            await asyncio.sleep(e.retry_after)
            raise
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((TelegramError, RetryAfter)),
        before_sleep=lambda retry_state: logger.debug(
            f"Retrying Telegram photo send after attempt {retry_state.attempt_number}"
        ),
    )
    async def _send_photo_with_retry(
        self, photo: InputFile, caption: str, parse_mode: Optional[str] = None
    ):
        """
        Sends a photo with retry logic.

        Args:
            photo (InputFile): The photo to send.
            caption (str): Caption for the photo.
            parse_mode (str, optional): Parse mode for the caption.
        """
        try:
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo,
                caption=caption,
                parse_mode=parse_mode,
            )
        except RetryAfter as e:
            self.logger.warning(
                f"Telegram rate limit hit, retrying after {e.retry_after} seconds"
            )
            await asyncio.sleep(e.retry_after)
            raise
        except TelegramError as e:
            self.logger.error(f"Telegram error: {e}")
            raise

    async def handle_callback(self, update, context):
        """Handles inline keyboard callbacks."""
        try:
            query = update.callback_query
            data = query.data
            if data == "portfolio":
                with open(config.config.PORTFOLIO_FILE, "r") as f:
                    portfolio = json.load(f)
                self.notify_portfolio_summary(portfolio)
            elif data == "trades":
                buy_trades = (
                    pd.read_csv(config.config.BUY_TRADES_CSV)
                    .tail(5)
                    .to_dict(orient="records")
                )
                finished_trades = (
                    pd.read_csv(config.config.FINISHED_TRADES_CSV)
                    .tail(5)
                    .to_dict(orient="records")
                )
                message = "<b>Recent Trades</b>\n"
                if buy_trades:
                    message += "\n<b>Buy Trades:</b>\n"
                    for trade in buy_trades:
                        message += (
                            f"{trade['Symbol']}: {float(trade['Buy Quantity']):.4f} @ €{float(trade['Buy Price']):.4f} "
                            f"({trade['Buy Time']}, Slippage: {trade['Buy Slippage']})\n"
                        )
                if finished_trades:
                    message += "\n<b>Finished Trades:</b>\n"
                    for trade in finished_trades:
                        message += (
                            f"{trade['Symbol']}: €{float(trade['Profit/Loss']):.2f} "
                            f"({trade['Sell Time']}, Reason: {trade['Reason']})\n"
                        )
                self.queue.put((message, True))
            elif data == "stats":
                finished_trades = pd.read_csv(config.config.FINISHED_TRADES_CSV)
                total_trades = len(finished_trades)
                win_rate = (
                    len(
                        finished_trades[
                            finished_trades["Profit/Loss"].astype(float) > 0
                        ]
                    )
                    / total_trades
                    * 100
                    if total_trades > 0
                    else 0
                )
                total_pl = finished_trades["Profit/Loss"].astype(float).sum()
                avg_pl = (
                    finished_trades["Profit/Loss"].astype(float).mean()
                    if total_trades > 0
                    else 0
                )
                message = (
                    "<b>Performance Stats</b>\n"
                    f"Total Trades: {total_trades}\n"
                    f"Win Rate: {win_rate:.2f}%\n"
                    f"Total Profit/Loss: €{total_pl:.2f}\n"
                    f"Average Profit/Loss: €{avg_pl:.2f}"
                )
                self.queue.put((message, True))
            elif data == "logs":
                log_file = "trading_bot.log"
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        lines = f.readlines()[-10:]  # Last 10 lines
                    message = "<b>Recent Logs</b>\n" + "".join(lines)
                    self.queue.put((message, False))
                else:
                    self.queue.put(("<b>Log file not found</b>", True))
            await query.answer()
        except Exception as e:
            self.logger.error(f"Error handling callback {data}: {e}", exc_info=True)
            await query.answer(text=f"Error: {e}")

    def notify_buy_trade(self, trade_data: Dict):
        """
        Queues a notification for a buy trade.

        Args:
            trade_data (Dict): Dictionary containing buy trade details.
        """
        try:
            message = (
                "<b>Buy Trade Executed</b>\n"
                f"Symbol: {trade_data['Symbol']}\n"
                f"Quantity: {float(trade_data['Buy Quantity']):.4f}\n"
                f"Price: €{float(trade_data['Buy Price']):.4f}\n"
                f"Time: {trade_data['Buy Time']}\n"
                f"Slippage: {trade_data['Buy Slippage']}\n"
                f"Cost: €{float(trade_data['Actual Cost']):.2f}\n"
                f"Fee: €{float(trade_data['Buy Fee']):.2f}\n"
                f"Trade Count: {trade_data['Trade Count']}\n"
                f"Largest Trade: €{float(trade_data['Largest Trade Volume EUR']):.2f}"
            )
            self.queue.put((message, True))
            self.logger.debug(
                f"Queued buy trade notification for {trade_data['Symbol']}"
            )
        except Exception as e:
            self.logger.error(
                f"Error queuing buy trade notification: {e}", exc_info=True
            )

    def notify_sell_trade(self, trade_data: Dict):
        """
        Queues a notification for a sell trade.

        Args:
            trade_data (Dict): Dictionary containing sell trade details.
        """
        try:
            profit_loss = float(trade_data["Profit/Loss"])
            message = (
                "<b>Sell Trade Executed</b>\n"
                f"Symbol: {trade_data['Symbol']}\n"
                f"Quantity: {float(trade_data['Sell Quantity']):.4f}\n"
                f"Sell Price: €{float(trade_data['Sell Price']):.4f}\n"
                f"Time: {trade_data['Sell Time']}\n"
                f"Slippage: {trade_data['Sell Slippage']}\n"
                f"Fee: €{float(trade_data['Sell Fee']):.2f}\n"
                f"Profit/Loss: €{profit_loss:.2f} ({'📈' if profit_loss >= 0 else '📉'}) \n"
                f"Reason: {trade_data['Reason']}"
            )
            self.queue.put((message, True))
            self.logger.debug(
                f"Queued sell trade notification for {trade_data['Symbol']}"
            )
        except Exception as e:
            self.logger.error(
                f"Error queuing sell trade notification: {e}", exc_info=True
            )

    def notify_portfolio_summary(self, portfolio: Dict):
        """
        Queues a notification for a portfolio summary.

        Args:
            portfolio (Dict): Portfolio data with cash, assets, and total value.
        """
        try:
            total_asset_value = sum(
                asset["quantity"] * asset["current_price"]
                for asset in portfolio.get("assets", {}).values()
            )
            total_value = portfolio.get("cash", 0) + total_asset_value
            message = (
                "<b>Portfolio Summary</b>\n"
                f"Cash: €{portfolio.get('cash', 0):.2f}\n"
                f"Assets: {len(portfolio.get('assets', {}))}\n"
                f"Total Value: €{total_value:.2f}\n"
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if portfolio.get("assets"):
                message += "\n<b>Assets:</b>\n"
                for symbol, asset in portfolio.get("assets", {}).items():
                    unrealized_profit = (
                        (asset["current_price"] - asset["purchase_price"])
                        / asset["purchase_price"]
                        if asset["purchase_price"] > 0
                        else 0
                    )
                    message += (
                        f"{symbol}: {asset['quantity']:.4f} @ €{asset['current_price']:.4f} "
                        f"(P/L: {unrealized_profit*100:.2f}%)\n"
                    )
            self.queue.put((message, True))
            self.logger.debug("Queued portfolio summary notification")
        except Exception as e:
            self.logger.error(
                f"Error queuing portfolio summary notification: {e}", exc_info=True
            )

    def notify_error(self, subject: str, message: str):
        """
        Queues an error notification.

        Args:
            subject (str): Error subject.
            message (str): Error message details.
        """
        try:
            message = f"<b>⚠️ Error: {subject}</b>\n{message}"
            self.queue.put((message, True))
            self.logger.debug(f"Queued error notification: {subject}")
        except Exception as e:
            self.logger.error(f"Error queuing error notification: {e}", exc_info=True)

    async def notify_performance_chart(self, portfolio_values: List[Dict]):
        """
        Sends a portfolio value chart as a PNG.

        Args:
            portfolio_values (List[Dict]): List of {'timestamp': str, 'portfolio_value': float}.
        """
        try:
            if not portfolio_values:
                self.queue.put(("<b>No portfolio data available for chart</b>", True))
                return
            times = [datetime.fromisoformat(v["timestamp"]) for v in portfolio_values]
            values = [v["portfolio_value"] for v in portfolio_values]
            plt.figure(figsize=(8, 4))
            plt.plot(times, values, label="Portfolio Value (€)")
            plt.xlabel("Time")
            plt.ylabel("Value (€)")
            plt.title("Portfolio Value Over Time")
            plt.legend()
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            self.queue.put(
                (
                    InputFile(buf, filename="portfolio.png"),
                    "<b>Portfolio Value Chart</b>",
                    True,
                )
            )
            plt.close()
            self.logger.debug("Queued portfolio chart")
        except Exception as e:
            self.logger.error(f"Error queuing portfolio chart: {e}", exc_info=True)
            self.queue.put((f"<b>Error generating chart: {e}</b>", True))

    async def notify_asset_allocation(self, portfolio: Dict):
        """
        Sends a pie chart of asset allocation.

        Args:
            portfolio (Dict): Portfolio data with assets.
        """
        try:
            assets = portfolio.get("assets", {})
            if not assets:
                self.queue.put(("<b>No assets for allocation chart</b>", True))
                return
            labels = []
            sizes = []
            for symbol, asset in assets.items():
                value = asset["quantity"] * asset["current_price"]
                labels.append(symbol)
                sizes.append(value)
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            plt.title("Portfolio Asset Allocation")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            self.queue.put(
                (
                    InputFile(buf, filename="allocation.png"),
                    "<b>Asset Allocation</b>",
                    True,
                )
            )
            plt.close()
            self.logger.debug("Queued asset allocation chart")
        except Exception as e:
            self.logger.error(
                f"Error queuing asset allocation chart: {e}", exc_info=True
            )
            self.queue.put((f"<b>Error generating chart: {e}</b>", True))

    async def update_pinned_summary(self, portfolio: Dict):
        """
        Updates the pinned portfolio summary message.

        Args:
            portfolio (Dict): Portfolio data with cash, assets, and total value.
        """
        try:
            total_asset_value = sum(
                asset["quantity"] * asset["current_price"]
                for asset in portfolio.get("assets", {}).values()
            )
            total_value = portfolio.get("cash", 0) + total_asset_value
            message = (
                "<b>Portfolio Summary (Pinned)</b>\n"
                f"Cash: €{portfolio.get('cash', 0):.2f}\n"
                f"Assets: {len(portfolio.get('assets', {}))}\n"
                f"Total Value: €{total_value:.2f}\n"
                f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if hasattr(self, "pinned_message_id"):
                try:
                    await self.bot.edit_message_text(
                        chat_id=self.chat_id,
                        message_id=self.pinned_message_id,
                        text=message,
                        parse_mode="HTML",
                    )
                    self.logger.debug("Updated pinned portfolio summary")
                    return
                except TelegramError as e:
                    self.logger.warning(f"Failed to update pinned message: {e}")
            # Pin a new message if none exists or update fails
            sent_message = await self.bot.send_message(
                chat_id=self.chat_id, text=message, parse_mode="HTML"
            )
            await self.bot.pin_chat_message(
                chat_id=self.chat_id,
                message_id=sent_message.message_id,
                disable_notification=True,
            )
            self.pinned_message_id = sent_message.message_id
            self.logger.debug("Pinned new portfolio summary")
        except Exception as e:
            self.logger.error(f"Error updating pinned summary: {e}", exc_info=True)
            self.queue.put((f"<b>Error updating pinned summary: {e}</b>", True))

    async def notify_performance_alert(
        self, portfolio: Dict, finished_trades: pd.DataFrame
    ):
        """
        Sends alerts for performance thresholds.

        Args:
            portfolio (Dict): Portfolio data with cash and assets.
            finished_trades (pd.DataFrame): DataFrame of finished trades.
        """
        try:
            total_asset_value = sum(
                asset["quantity"] * asset["current_price"]
                for asset in portfolio.get("assets", {}).values()
            )
            total_value = portfolio.get("cash", 0) + total_asset_value
            initial_value = config.config.PORTFOLIO_VALUE
            value_change = (
                (total_value - initial_value) / initial_value * 100
                if initial_value > 0
                else 0
            )
            if abs(value_change) >= 5:
                message = (
                    f"<b>Portfolio Alert</b>\n"
                    f"Portfolio value changed by {value_change:.2f}% (€{total_value:.2f})\n"
                    f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.queue.put((message, True))
            if not finished_trades.empty:
                recent_trade = finished_trades.iloc[-1]
                pl = float(recent_trade["Profit/Loss"])
                if pl <= -100:
                    message = (
                        f"<b>Trade Loss Alert</b>\n"
                        f"Symbol: {recent_trade['Symbol']}\n"
                        f"Loss: €{pl:.2f}\n"
                        f"Time: {recent_trade['Sell Time']}"
                    )
                    self.queue.put((message, True))
                win_rate = (
                    len(
                        finished_trades[
                            finished_trades["Profit/Loss"].astype(float) > 0
                        ]
                    )
                    / len(finished_trades)
                    * 100
                )
                if win_rate < 50 and len(finished_trades) >= 10:
                    message = (
                        f"<b>Performance Alert</b>\n"
                        f"Win Rate dropped to {win_rate:.2f}% ({len(finished_trades)} trades)\n"
                        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    self.queue.put((message, True))
            self.logger.debug("Checked performance thresholds")
        except Exception as e:
            self.logger.error(
                f"Error checking performance thresholds: {e}", exc_info=True
            )
            self.queue.put((f"<b>Error checking performance thresholds: {e}</b>", True))

    async def notify_daily_report(self):
        """
        Sends a daily performance report.
        """
        try:
            finished_trades = pd.read_csv(config.config.FINISHED_TRADES_CSV)
            total_trades = len(finished_trades)
            win_rate = (
                len(finished_trades[finished_trades["Profit/Loss"].astype(float) > 0])
                / total_trades
                * 100
                if total_trades > 0
                else 0
            )
            total_pl = finished_trades["Profit/Loss"].astype(float).sum()
            with open(config.config.PORTFOLIO_FILE, "r") as f:
                portfolio = json.load(f)
            total_value = portfolio.get("cash", 0) + sum(
                asset["quantity"] * asset["current_price"]
                for asset in portfolio.get("assets", {}).values()
            )
            message = (
                "<b>Daily Performance Report</b>\n"
                f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}\n"
                f"Portfolio Value: €{total_value:.2f}\n"
                f"Total Trades: {total_trades}\n"
                f"Win Rate: {win_rate:.2f}%\n"
                f"Total Profit/Loss: €{total_pl:.2f}"
            )
            self.queue.put((message, True))
            self.logger.debug("Queued daily report")
        except Exception as e:
            self.logger.error(f"Error queuing daily report: {e}", exc_info=True)
            self.queue.put((f"<b>Error generating daily report: {e}</b>", True))
