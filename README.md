# Crypto Trading Automation System

A comprehensive cryptocurrency trading bot that monitors whale transactions, analyzes market conditions, and executes automated trades on Solana/PumpFun.

## Features

- **Whale Transaction Monitoring**: Real-time tracking of large transactions on PumpFun
- **DEX Integration**: Monitors Raydium, Orca, and other DEX platforms
- **Multi-Strategy Trading**: Combines multiple technical indicators and strategies
- **AI-Powered Analysis**: Optional OpenAI integration for market sentiment analysis
- **Demo Mode**: Safe paper trading for testing strategies
- **Telegram Bot**: Real-time notifications and portfolio management
- **Risk Management**: Automated stop-loss and take-profit execution
- **Technical Analysis**: RSI, MACD, Moving Averages, Volume Analysis

## Project Structure

```
TradeAutomation/
├── src/
│   ├── config.py              # Configuration management
│   ├── scrapers/
│   │   ├── pumpfun.py         # PumpFun whale monitoring
│   │   ├── dex.py             # DEX activity tracking
│   │   └── market.py          # Market data collection
│   ├── analyzers/
│   │   ├── ai_analyzer.py     # AI-powered analysis
│   │   └── strategies.py      # Trading strategies
│   ├── traders/
│   │   ├── demo_trader.py     # Paper trading
│   │   └── live_trader.py     # Live trading (use with caution)
│   ├── notifications/
│   │   └── telegram_bot.py    # Telegram integration
│   └── database/
│       ├── models.py           # Database models
│       └── manager.py          # Database operations
├── main.py                     # Main orchestrator
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker setup
├── Dockerfile                  # Container configuration
├── .env.example                # Environment template
└── README.md                   # This file
```

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)
- Telegram Bot Token
- Solana RPC URL (e.g., QuickNode, Helius)
- OpenAI API Key (optional, for AI analysis)

## Installation

### Option 1: Local Setup

1. **Clone and setup environment**:
```bash
cd TradeAutomation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
nano .env
```

4. **Start database services**:
```bash
docker-compose up -d postgres redis
```

5. **Run the bot**:
```bash
python main.py
```

### Option 2: Docker Setup

1. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
nano .env
```

2. **Start all services**:
```bash
docker-compose up -d
```

3. **View logs**:
```bash
docker-compose logs -f bot
```

## Configuration

Edit `.env` file with your settings:

```env
# Required
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Database
DATABASE_URL=postgresql://cryptouser:password@localhost/cryptobot
DB_PASSWORD=your_secure_password

# Trading (start in demo mode!)
DEMO_MODE=true
INITIAL_BALANCE=1000
WHALE_THRESHOLD_SOL=10

# Risk Management
MAX_POSITION_SIZE=0.1        # 10% per position
STOP_LOSS_PERCENT=5          # 5% stop loss
TAKE_PROFIT_PERCENT=15       # 15% take profit

# Optional - AI Analysis
ENABLE_AI_ANALYSIS=false
OPENAI_API_KEY=your_openai_key
```

## Telegram Bot Setup

1. **Create a bot**:
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Use `/newbot` command
   - Copy the bot token

2. **Get your chat ID**:
   - Message [@userinfobot](https://t.me/userinfobot)
   - Copy your ID

3. **Add credentials to `.env`**:
```env
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=123456789
```

## Telegram Commands

- `/start` - Initialize the bot
- `/status` - Show bot status
- `/portfolio` - View current positions and PnL
- `/pause` - Pause trading
- `/resume` - Resume trading
- `/help` - Show available commands

## Trading Strategies

The bot uses a multi-strategy approach:

1. **Moving Average Crossover**: Fast/slow MA signals
2. **RSI Momentum**: Oversold/overbought detection
3. **Volume Profile**: Volume surge analysis
4. **Whale Follower**: Track smart money movements

Strategies are weighted and combined for final decisions.

## Safety Features

### Demo Mode (Default)
- Paper trading with virtual balance
- No real money risk
- Full feature testing
- Trade history tracking

### Risk Management
- Position size limits (default 10% max)
- Automatic stop-loss (default 5%)
- Automatic take-profit (default 15%)
- Configurable risk parameters

### Live Trading Safeguards
⚠️ **IMPORTANT**: Live trading is intentionally incomplete and requires additional implementation for your safety.

Before enabling live trading:
1. Test in demo mode for at least 30 days
2. Implement secure wallet key management
3. Start with small amounts you can afford to lose
4. Monitor continuously
5. Never leave unattended

## Monitoring

### Logs
```bash
# Local
tail -f logs/bot_*.log

# Docker
docker-compose logs -f bot
```

### Reports
- Daily reports sent via Telegram
- Trade history exported to `reports/`
- Portfolio snapshots available

### Alerts
- 🐋 Whale transactions
- ✅ Trade executions
- 📊 Position updates
- ⚠️ Risk warnings

## Database

The bot uses PostgreSQL to store:
- Whale alerts
- Market events
- Trade history
- System logs

### Manual Database Access
```bash
docker exec -it tradeautomation-postgres-1 psql -U cryptouser -d cryptobot
```

## Performance Optimization

### Solana RPC
For production, use a premium RPC provider:
- [QuickNode](https://www.quicknode.com/)
- [Helius](https://www.helius.dev/)
- [Alchemy](https://www.alchemy.com/)

Free RPC endpoints may have rate limits.

### Redis Caching
Redis is used for:
- Price caching
- Session management
- Rate limiting

## Troubleshooting

### Bot won't start
```bash
# Check logs
tail -f logs/bot_*.log

# Verify database connection
docker-compose ps

# Check environment variables
cat .env
```

### No whale alerts
- Verify WebSocket connection in logs
- Check `WHALE_THRESHOLD_SOL` setting
- Ensure PumpFun API is accessible

### Telegram not working
- Verify bot token and chat ID
- Check if bot is started with `/start` command
- Review Telegram-related logs

## Development

### Adding New Strategies
1. Create strategy class in `src/analyzers/strategies.py`
2. Inherit from `Strategy` base class
3. Implement `analyze()` method
4. Add to `StrategyManager`

### Adding New Scrapers
1. Create scraper in `src/scrapers/`
2. Implement data collection methods
3. Register in `main.py`
4. Update configuration if needed

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Backtest Mode

Enable backtesting in `.env`:
```env
ENABLE_BACKTESTING=true
```

This allows testing strategies against historical data before live deployment.

## Roadmap

- [ ] Additional DEX integrations (Jupiter, Orca)
- [ ] Enhanced ML models for predictions
- [ ] Web dashboard for monitoring
- [ ] Mobile app for alerts
- [ ] Support for multiple chains
- [ ] Advanced backtesting engine
- [ ] Social sentiment analysis
- [ ] Portfolio rebalancing

## Security Best Practices

1. **Never commit `.env` file**
2. **Use strong database passwords**
3. **Enable 2FA on all accounts**
4. **Regularly update dependencies**
5. **Monitor for suspicious activity**
6. **Keep API keys secure**
7. **Use hardware wallets for live trading**
8. **Regular security audits**

## Disclaimer

⚠️ **CRITICAL WARNING**:
- Cryptocurrency trading is extremely risky
- You can lose all your invested capital
- Past performance does not guarantee future results
- This bot is provided as-is without warranties
- Use at your own risk
- Always start with demo mode
- Never invest more than you can afford to lose
- This is not financial advice

## Support

For issues and questions:
- Check logs first
- Review this documentation
- Search existing GitHub issues
- Open a new issue with details

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

Built with:
- Python & AsyncIO
- Solana SDK
- python-telegram-bot
- SQLAlchemy
- Technical Analysis library (ta)
- OpenAI API (optional)

---

**Remember**: ALWAYS start in DEMO_MODE=true and test thoroughly before considering live trading!
