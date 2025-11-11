# Quick Start Guide

Get your crypto trading bot running in 5 minutes!

## Step 1: Setup Environment (2 min)

```bash
# Copy environment template
cp .env.example .env

# Edit with your details
nano .env
```

Required settings:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DEMO_MODE=true
```

## Step 2: Install Dependencies (2 min)

### Option A: Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Docker (Recommended)
```bash
docker-compose up -d
```

## Step 3: Start the Bot (1 min)

### Python:
```bash
python main.py
```

### Docker:
```bash
docker-compose logs -f bot
```

## Step 4: Verify

Check Telegram - you should receive:
```
🚀 Bot initialized and starting up...
```

## Common Issues

### "TELEGRAM_BOT_TOKEN not found"
- Create bot with [@BotFather](https://t.me/botfather)
- Add token to `.env`

### "Database connection failed"
```bash
docker-compose up -d postgres redis
```

### "Module not found"
```bash
pip install -r requirements.txt
```

## Next Steps

1. Send `/help` to your bot
2. Monitor whale alerts
3. Check `/portfolio` regularly
4. Test for 30+ days in demo mode
5. Read full README.md

## Important

- ⚠️ ALWAYS keep `DEMO_MODE=true` for testing
- Never commit `.env` file
- Start with small whale thresholds
- Monitor logs regularly

## Get Help

```bash
# View logs
tail -f logs/bot_*.log

# Check status
docker-compose ps

# Restart bot
docker-compose restart bot
```

Happy Trading! 🚀
