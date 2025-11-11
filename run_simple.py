"""
Simplified runner - No database required for initial testing
"""
import asyncio
from datetime import datetime
from loguru import logger
import os

# Set environment to skip database
os.environ['SKIP_DATABASE'] = 'true'

from src.config import config
from src.scrapers.market import MarketDataCollector
from src.analyzers.ai_analyzer import AIAnalyzer
from src.analyzers.strategies import StrategyManager
from src.traders.demo_trader import DemoTrader
from src.notifications.telegram_bot import TelegramBot

class SimpleCryptoBot:
    """Simplified bot without database dependencies"""

    def __init__(self):
        logger.info("Initializing Simple Crypto Trading Bot...")

        # Initialize components (no database)
        self.market_collector = MarketDataCollector()
        self.ai_analyzer = AIAnalyzer()
        self.strategy_manager = StrategyManager()
        self.trader = DemoTrader(config.INITIAL_BALANCE)
        self.telegram = TelegramBot(trader=self.trader, analyzer=self.ai_analyzer)

        logger.info(f"Running in {'DEMO' if config.DEMO_MODE else 'LIVE'} mode")
        logger.info(f"Initial balance: ${config.INITIAL_BALANCE}")

    async def test_system(self):
        """Test system components"""
        try:
            logger.info("Testing Telegram bot...")
            await self.telegram.initialize()
            await self.telegram.send_alert("🚀 Simple Bot Started!\n\nTesting system components...")

            # Test portfolio
            portfolio = self.trader.get_portfolio_status()
            logger.info(f"Portfolio initialized: ${portfolio['cash']}")

            await self.telegram.send_alert(
                f"✅ System Ready!\n\n"
                f"💰 Balance: ${portfolio['cash']:.2f}\n"
                f"📊 Mode: {'DEMO' if config.DEMO_MODE else 'LIVE'}\n\n"
                f"Send /help for commands"
            )

            # Start polling for commands
            logger.info("Bot is running and listening for commands. Press Ctrl+C to stop.")
            logger.info("Try /help in Telegram!")

            # Start the bot polling
            await self.telegram.app.updater.start_polling()

            # Keep running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await self.telegram.send_alert("🛑 Bot stopped")
        except Exception as e:
            logger.error(f"Error: {e}")
            await self.telegram.send_alert(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    logger.add(
        "logs/simple_bot_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )

    bot = SimpleCryptoBot()
    await bot.test_system()

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE CRYPTO TRADING BOT")
    print("No database required - Perfect for testing!")
    print("=" * 60)
    asyncio.run(main())
