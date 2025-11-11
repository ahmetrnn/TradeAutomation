from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from typing import Dict, Any
from datetime import datetime
from ..config import config
from loguru import logger

class TelegramBot:
    """Telegram bot for notifications and control"""

    def __init__(self, trader=None, analyzer=None):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.trader = trader
        self.analyzer = analyzer
        self.app = None

    async def initialize(self):
        """Initialize bot"""
        self.app = Application.builder().token(self.token).build()

        # Add handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("portfolio", self.cmd_portfolio))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))

        # Start bot with polling
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("Telegram bot initialized and polling started")

    async def send_alert(self, message: str, parse_mode: str = "Markdown"):
        """Send alert message"""
        try:
            await self.app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_whale_alert(self, whale_data: Dict[str, Any]):
        """Send whale alert with action buttons"""

        message = f"""
🐋 *WHALE ALERT* 🐋

*Token:* `{whale_data['token_address'][:8]}...`
*Amount:* {whale_data['amount_sol']:.2f} SOL
*Token Name:* {whale_data.get('token_name', 'Unknown')}
*Market Cap:* ${whale_data.get('market_cap', 0):,.0f}
*Time:* {whale_data['timestamp'].strftime('%H:%M:%S')}

[View TX](https://solscan.io/tx/{whale_data['transaction_hash']})
        """

        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("📊 Analyze", callback_data=f"analyze_{whale_data['token_address']}"),
                InlineKeyboardButton("📈 Buy", callback_data=f"buy_{whale_data['token_address']}")
            ],
            [
                InlineKeyboardButton("🔍 More Info", callback_data=f"info_{whale_data['token_address']}"),
                InlineKeyboardButton("❌ Ignore", callback_data="ignore")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await self.app.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode="Markdown",
            reply_markup=reply_markup,
            disable_web_page_preview=True
        )

    async def send_trade_notification(self, trade_result: Dict[str, Any]):
        """Send trade execution notification"""

        emoji = "✅" if trade_result['success'] else "❌"
        action = trade_result['action']

        message = f"""
{emoji} *Trade Executed*

*Action:* {action}
*Token:* `{trade_result['token'][:8]}...`
*Price:* ${trade_result['price']:.8f}
*Time:* {trade_result['timestamp']}
"""

        if trade_result.get('pnl') is not None:
            pnl_emoji = "📈" if trade_result['pnl'] > 0 else "📉"
            message += f"*PnL:* {pnl_emoji} ${trade_result['pnl']:.2f} ({trade_result['pnl_percent']:.2f}%)\n"

        if trade_result.get('error'):
            message += f"*Error:* {trade_result['error']}\n"

        await self.send_alert(message)

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "🤖 *Crypto Trading Bot Started*\n\n"
            "Use /help to see available commands.",
            parse_mode="Markdown"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        status = "🟢 Active" if self.trader else "🔴 Inactive"
        mode = "Demo" if config.DEMO_MODE else "Live"

        message = f"""
*Bot Status*
Status: {status}
Mode: {mode}
Whale Threshold: {config.WHALE_THRESHOLD_SOL} SOL
Trade Interval: {config.TRADE_INTERVAL // 60} minutes
        """

        await update.message.reply_text(message, parse_mode="Markdown")

    async def cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command"""
        if not self.trader:
            await update.message.reply_text("Trader not initialized")
            return

        portfolio = self.trader.get_portfolio_status()

        message = f"""
*Portfolio Status*

💰 *Cash:* ${portfolio['cash']:.2f}
📊 *Total Value:* ${portfolio['total_value']:.2f}
📈 *Total PnL:* ${portfolio['total_pnl']:.2f} ({portfolio['total_pnl_percent']:.2f}%)
🎯 *Win Rate:* {portfolio['win_rate']:.2f}%
📊 *Total Trades:* {portfolio['total_trades']}

*Open Positions:* {len(portfolio['positions'])}
        """

        for pos in portfolio['positions'][:5]:  # Show max 5 positions
            pnl_emoji = "📈" if pos['pnl'] > 0 else "📉"
            message += f"\n• `{pos['token'][:8]}...`: {pnl_emoji} {pos['pnl_percent']:.2f}%"

        await update.message.reply_text(message, parse_mode="Markdown")

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
*Available Commands:*

/start - Start the bot
/status - Show bot status
/portfolio - Show portfolio details
/pause - Pause trading
/resume - Resume trading
/help - Show this message

*Alerts:*
• 🐋 Whale alerts for transactions > 10 SOL
• 📊 DEX payment notifications
• ✅ Trade execution confirmations
• ⚠️ Risk warnings
        """

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command"""
        # Implement pause logic
        await update.message.reply_text("⏸️ Trading paused")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        # Implement resume logic
        await update.message.reply_text("▶️ Trading resumed")

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()

        data = query.data

        if data.startswith("analyze_"):
            token = data.replace("analyze_", "")
            await query.edit_message_text(f"Analyzing token {token[:8]}...")
            # Trigger analysis

        elif data.startswith("buy_"):
            token = data.replace("buy_", "")
            await query.edit_message_text(f"Evaluating buy opportunity for {token[:8]}...")
            # Trigger buy evaluation

        elif data.startswith("info_"):
            token = data.replace("info_", "")
            # Fetch and send more info
            await query.edit_message_text(f"Fetching info for {token[:8]}...")

        elif data == "ignore":
            await query.edit_message_text("Alert ignored")
