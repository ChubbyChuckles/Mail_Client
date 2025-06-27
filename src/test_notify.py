# test_notify.py
import asyncio
from .telegram_notifications import TelegramNotifier
from .config import config  # Adjust import based on your config file

async def test_notifier():
    notifier = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID
    )
    try:
        # Start the notifier's polling in a background task
        polling_task = asyncio.create_task(notifier.start())
        # Queue a test notification
        notifier.notify_error("Test Notification", "Testing Telegram integration")
        # Wait for the notification to be processed
        await asyncio.sleep(5)
        # Send another test notification to confirm queue processing
        notifier.notify_error("Second Test", "Confirming queue processing")
        await asyncio.sleep(5)
    finally:
        # Stop the notifier
        notifier.running = False  # Stop the queue processing loop
        await notifier.stop()  # Stop the Telegram application
        polling_task.cancel()  # Cancel the polling task
        try:
            await polling_task  # Ensure the task is fully cancelled
        except asyncio.CancelledError:
            pass

def run_test():
    # Use a new event loop for the test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_notifier())
    finally:
        # Clean up the event loop
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

if __name__ == "__main__":
    run_test()