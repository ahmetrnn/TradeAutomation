import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Centralized configuration management"""

    # API Keys
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Solana/PumpFun Configuration
    SOLANA_RPC_URL: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    PUMPFUN_WS_URL: str = os.getenv("PUMPFUN_WS_URL", "wss://pumpfun.com/api/ws")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/cryptobot")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Trading Parameters
    WHALE_THRESHOLD_SOL: float = float(os.getenv("WHALE_THRESHOLD_SOL", "10"))
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    INITIAL_BALANCE: float = float(os.getenv("INITIAL_BALANCE", "1000"))

    # Risk Management
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))  # 10% of portfolio
    MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "5"))  # Max 5 positions at once
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "5"))
    TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "15"))
    MAX_DAILY_LOSS: float = float(os.getenv("MAX_DAILY_LOSS", "100"))  # Max $100 loss per day
    MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "20"))  # Stop if down 20%

    # Intervals (in seconds)
    WHALE_CHECK_INTERVAL: int = int(os.getenv("WHALE_CHECK_INTERVAL", "30"))
    TRADE_INTERVAL: int = int(os.getenv("TRADE_INTERVAL", "900"))  # 15 minutes

    # Feature Flags
    ENABLE_AI_ANALYSIS: bool = os.getenv("ENABLE_AI_ANALYSIS", "false").lower() == "true"
    ENABLE_BACKTESTING: bool = os.getenv("ENABLE_BACKTESTING", "true").lower() == "true"

config = Config()
