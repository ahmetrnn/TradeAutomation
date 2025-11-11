import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

# Import all components
from src.config import config
from src.database.manager import db
from src.scrapers.pumpfun import PumpFunMonitor
from src.scrapers.dex import DEXMonitor
from src.scrapers.market import MarketDataCollector
from src.analyzers.ai_analyzer import AIAnalyzer
from src.analyzers.strategies import StrategyManager
from src.traders.demo_trader import DemoTrader
from src.traders.live_trader import LiveTrader
from src.notifications.telegram_bot import TelegramBot

class CryptoTradingBot:
    """Main bot orchestrator"""

    def __init__(self):
        self.running = False
        self.scheduler = AsyncIOScheduler()

        # Initialize components
        self.pumpfun_monitor = None
        self.dex_monitor = DEXMonitor()
        self.market_collector = MarketDataCollector()
        self.ai_analyzer = AIAnalyzer()
        self.strategy_manager = StrategyManager()

        # Initialize trader based on mode
        if config.DEMO_MODE:
            self.trader = DemoTrader(config.INITIAL_BALANCE)
            logger.info("Running in DEMO mode")
        else:
            self.trader = LiveTrader()
            logger.warning("Running in LIVE mode - Real money at risk!")

        # Initialize Telegram bot
        self.telegram = TelegramBot(
            trader=self.trader,
            analyzer=self.ai_analyzer
        )

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Crypto Trading Bot...")

        # Initialize Telegram bot
        await self.telegram.initialize()
        await self.telegram.send_alert("🚀 Bot initialized and starting up...")

        # Initialize PumpFun monitor with whale callback
        self.pumpfun_monitor = PumpFunMonitor(
            on_whale_detected=self.handle_whale_alert
        )

        # Schedule regular tasks
        self.scheduler.add_job(
            self.trading_cycle,
            'interval',
            seconds=config.TRADE_INTERVAL,
            id='trading_cycle'
        )

        self.scheduler.add_job(
            self.check_positions,
            'interval',
            seconds=60,  # Check every minute
            id='position_check'
        )

        self.scheduler.add_job(
            self.generate_report,
            'cron',
            hour=0,  # Daily at midnight
            id='daily_report'
        )

        logger.info("Initialization complete")

    async def handle_whale_alert(self, whale_data: Dict[str, Any]):
        """Handle detected whale transaction"""
        try:
            # Save to database
            db.save_whale_alert(whale_data)

            # Send Telegram alert
            await self.telegram.send_whale_alert(whale_data)

            # Analyze token
            token_address = whale_data['token_address']

            # Check DEX status
            dex_info = await self.dex_monitor.check_dex_payment(token_address)

            if dex_info['is_paid']:
                # Monitor post-DEX activity
                post_dex_analysis = await self.dex_monitor.monitor_post_dex_activity(
                    token_address,
                    duration_hours=1
                )

                if post_dex_analysis['risk_score'] < 70:
                    # Consider for trading
                    await self.evaluate_trading_opportunity(token_address, whale_data)

        except Exception as e:
            logger.error(f"Error handling whale alert: {e}")

    async def evaluate_trading_opportunity(
        self,
        token_address: str,
        whale_data: Dict[str, Any]
    ):
        """Evaluate if we should trade this token"""
        try:
            # Collect market data
            market_data = await self.market_collector.get_token_data(token_address)

            # Get AI analysis
            ai_analysis = await self.ai_analyzer.analyze_market_conditions(market_data)

            # Get strategy signals
            strategy_signal = await self.strategy_manager.get_combined_signal(market_data)

            # Combine analyses
            if (ai_analysis.action in ['BUY', 'SELL'] and
                strategy_signal['confidence'] > 0.4):

                # Execute trade
                current_price = market_data.get('price', 0)
                trade_result = await self.trader.execute_trade(
                    strategy_signal,
                    current_price,
                    token_address
                )

                # Notify
                await self.telegram.send_trade_notification(trade_result)

        except Exception as e:
            logger.error(f"Error evaluating opportunity: {e}")

    async def trading_cycle(self):
        """Regular trading cycle (15 minutes)"""
        try:
            logger.info("Starting trading cycle...")

            # Get current positions
            portfolio = self.trader.get_portfolio_status()

            # Check existing positions
            for position in portfolio['positions']:
                token = position['token']
                market_data = await self.market_collector.get_token_data(token)

                # Update current prices
                current_prices = {token: market_data.get('price', position['current_price'])}

                # Check stop loss / take profit
                await self.trader.check_stop_loss_take_profit(current_prices)

                # Get new signals
                signal = await self.strategy_manager.get_combined_signal(market_data)

                if signal['action'] == 'SELL' and signal['confidence'] > 0.4:
                    trade_result = await self.trader.execute_trade(
                        signal,
                        current_prices[token],
                        token
                    )
                    await self.telegram.send_trade_notification(trade_result)

            logger.info("Trading cycle complete")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    async def check_positions(self):
        """Regular position monitoring"""
        try:
            portfolio = self.trader.get_portfolio_status()

            # Alert if significant PnL
            for position in portfolio['positions']:
                if abs(position['pnl_percent']) > 10:
                    emoji = "📈" if position['pnl_percent'] > 0 else "📉"
                    message = f"{emoji} Position Update: {position['token'][:8]}... {position['pnl_percent']:.2f}%"
                    await self.telegram.send_alert(message)

        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    async def generate_report(self):
        """Generate daily report"""
        try:
            portfolio = self.trader.get_portfolio_status()
            recent_alerts = db.get_recent_whale_alerts(24)

            report = f"""
📊 *Daily Report*

*Portfolio Performance:*
• Total Value: ${portfolio['total_value']:.2f}
• Daily PnL: ${portfolio['total_pnl']:.2f}
• Win Rate: {portfolio['win_rate']:.2f}%
• Total Trades: {portfolio['total_trades']}

*Activity Summary:*
• Whale Alerts: {len(recent_alerts)}
• Open Positions: {len(portfolio['positions'])}

Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
            """

            await self.telegram.send_alert(report)

            # Export trade history
            self.trader.export_trade_history(
                f"reports/trade_history_{datetime.utcnow().strftime('%Y%m%d')}.json"
            )

        except Exception as e:
            logger.error(f"Error generating report: {e}")

    async def run(self):
        """Main run loop"""
        self.running = True

        try:
            # Initialize components
            await self.initialize()

            # Start scheduler
            self.scheduler.start()

            # Start PumpFun monitor
            monitor_task = asyncio.create_task(self.pumpfun_monitor.start())

            # Keep running
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.telegram.send_alert(f"❌ Bot crashed: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup on shutdown"""
        logger.info("Cleaning up...")

        # Stop components
        if self.pumpfun_monitor:
            self.pumpfun_monitor.stop()

        # Stop scheduler
        self.scheduler.shutdown()

        # Send final notification
        await self.telegram.send_alert("🛑 Bot shutting down")

        # Export final trade history
        self.trader.export_trade_history("reports/final_trade_history.json")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Configure logging
    logger.add(
        "logs/bot_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and run bot
    bot = CryptoTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
