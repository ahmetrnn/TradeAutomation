# Crypto Trading Automation System - Python Implementation (No N8N)

## Executive Summary
This document provides a complete implementation plan for a cryptocurrency trading automation system using Python, without relying on N8N. The system includes whale tracking, market analysis, automated trading, and Telegram notifications - all built with open-source tools.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON ORCHESTRATOR                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Scrapers   │  │   Analyzers  │  │   Executors  │     │
│  │              │  │              │  │              │     │
│  │ • WebSocket  │──│ • AI Engine  │──│ • Telegram   │     │
│  │ • REST APIs  │  │ • Strategies │  │ • Trading    │     │
│  │ • Blockchain │  │ • Signals    │  │ • Database   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Scheduler (APScheduler/Celery)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Complete Project Structure

```
crypto-trading-bot/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── main.py             # Main orchestrator
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── pumpfun.py      # PumpFun scraper
│   │   ├── dex.py          # DEX monitoring
│   │   └── market.py       # Market data collection
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── whale_detector.py
│   │   ├── ai_analyzer.py
│   │   └── strategies.py
│   ├── traders/
│   │   ├── __init__.py
│   │   ├── demo_trader.py
│   │   └── live_trader.py
│   ├── notifications/
│   │   ├── __init__.py
│   │   └── telegram_bot.py
│   └── database/
│       ├── __init__.py
│       ├── models.py
│       └── manager.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

## Phase 1: Environment Setup

### 1.1 Requirements File (requirements.txt)

```txt
# Core Dependencies
python-dotenv==1.0.0
requests==2.31.0
aiohttp==3.9.0
websockets==12.0

# Blockchain & Crypto
web3==6.11.0
solana==0.30.2
solders==0.18.1

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1

# Scheduling & Async
celery==5.3.4
apscheduler==3.10.4
asyncio==3.4.3

# Data Processing
pandas==2.1.3
numpy==1.24.3
ta==0.10.2  # Technical Analysis

# Machine Learning (optional)
scikit-learn==1.3.2
tensorflow==2.15.0  # or pytorch

# Notifications
python-telegram-bot==20.7
discord.py==2.3.2

# Web Scraping
beautifulsoup4==4.12.2
selenium==4.15.2
playwright==1.40.0

# API Clients
ccxt==4.1.52  # Crypto exchange library
pycoingecko==3.1.0

# Monitoring & Logging
loguru==0.7.2
sentry-sdk==1.39.1
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
```

### 1.2 Configuration Management (config.py)

```python
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
    STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "5"))
    TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "15"))
    
    # Intervals (in seconds)
    WHALE_CHECK_INTERVAL: int = int(os.getenv("WHALE_CHECK_INTERVAL", "30"))
    TRADE_INTERVAL: int = int(os.getenv("TRADE_INTERVAL", "900"))  # 15 minutes
    
    # Feature Flags
    ENABLE_AI_ANALYSIS: bool = os.getenv("ENABLE_AI_ANALYSIS", "false").lower() == "true"
    ENABLE_BACKTESTING: bool = os.getenv("ENABLE_BACKTESTING", "true").lower() == "true"

config = Config()
```

## Phase 2: Database Models

### 2.1 SQLAlchemy Models (database/models.py)

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class WhaleAlert(Base):
    __tablename__ = 'whale_alerts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_address = Column(String(64), nullable=False)
    whale_address = Column(String(64), nullable=False)
    amount_sol = Column(Float, nullable=False)
    transaction_hash = Column(String(128), unique=True)
    alert_sent = Column(Boolean, default=False)
    metadata = Column(JSON, default=dict)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'token_address': self.token_address,
            'whale_address': self.whale_address,
            'amount_sol': self.amount_sol,
            'tx_hash': self.transaction_hash,
            'metadata': self.metadata
        }

class MarketEvent(Base):
    __tablename__ = 'market_events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_address = Column(String(64), nullable=False)
    event_type = Column(String(50), nullable=False)  # 'dex_paid', 'whale_sell', 'whale_buy'
    data = Column(JSON, default=dict)
    processed = Column(Boolean, default=False)

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_address = Column(String(64), nullable=False)
    action = Column(String(10))  # 'BUY', 'SELL'
    price = Column(Float)
    amount = Column(Float)
    total_value = Column(Float)
    pnl = Column(Float, nullable=True)
    is_demo = Column(Boolean, default=True)
    strategy_used = Column(String(50))
    metadata = Column(JSON, default=dict)

class SystemLog(Base):
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20))  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(50))
    message = Column(String(500))
    details = Column(JSON, default=dict)
```

### 2.2 Database Manager (database/manager.py)

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from .models import Base
from config import config
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            config.DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def save_whale_alert(self, alert_data):
        with self.get_session() as session:
            from .models import WhaleAlert
            alert = WhaleAlert(**alert_data)
            session.add(alert)
            return alert.id
    
    def get_recent_whale_alerts(self, hours=24):
        with self.get_session() as session:
            from .models import WhaleAlert
            from datetime import datetime, timedelta
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            alerts = session.query(WhaleAlert)\
                .filter(WhaleAlert.timestamp > cutoff)\
                .order_by(WhaleAlert.timestamp.desc())\
                .all()
            return [alert.to_dict() for alert in alerts]

db = DatabaseManager()
```

## Phase 3: PumpFun & Blockchain Monitoring

### 3.1 PumpFun Scraper (scrapers/pumpfun.py)

```python
import asyncio
import json
import websockets
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from loguru import logger
from config import config

class PumpFunMonitor:
    """Monitors PumpFun for whale transactions"""
    
    def __init__(self, on_whale_detected: Optional[Callable] = None):
        self.ws_url = config.PUMPFUN_WS_URL
        self.solana_client = AsyncClient(config.SOLANA_RPC_URL)
        self.on_whale_detected = on_whale_detected
        self.whale_threshold = config.WHALE_THRESHOLD_SOL
        self.running = False
        
    async def connect_websocket(self):
        """Establish WebSocket connection to PumpFun"""
        try:
            async with websockets.connect(self.ws_url) as websocket:
                logger.info("Connected to PumpFun WebSocket")
                
                # Subscribe to transaction feed
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "channel": "transactions"
                }))
                
                self.running = True
                while self.running:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=30.0
                        )
                        await self.process_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            await asyncio.sleep(5)  # Retry after 5 seconds
            if self.running:
                await self.connect_websocket()
    
    async def process_message(self, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "transaction":
                transaction = data.get("data", {})
                
                # Check if it's a buy transaction
                if transaction.get("instruction_type") == "buy":
                    amount_sol = transaction.get("sol_amount", 0)
                    
                    # Check for whale activity
                    if amount_sol >= self.whale_threshold:
                        await self.handle_whale_transaction(transaction)
                        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
    
    async def handle_whale_transaction(self, transaction: Dict[str, Any]):
        """Handle detected whale transaction"""
        whale_data = {
            "token_address": transaction.get("token_address"),
            "whale_address": transaction.get("buyer_address"),
            "amount_sol": transaction.get("sol_amount"),
            "transaction_hash": transaction.get("signature"),
            "timestamp": datetime.utcnow(),
            "token_name": transaction.get("token_name", "Unknown"),
            "price_per_token": transaction.get("price", 0),
            "market_cap": transaction.get("market_cap", 0)
        }
        
        logger.info(f"🐋 Whale detected: {whale_data['amount_sol']} SOL on {whale_data['token_name']}")
        
        if self.on_whale_detected:
            await self.on_whale_detected(whale_data)
    
    async def get_token_holders(self, token_address: str) -> list:
        """Get top holders for a token"""
        try:
            # This would require specific PumpFun API endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.pumpfun.com/tokens/{token_address}/holders"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("holders", [])
        except Exception as e:
            logger.error(f"Failed to fetch holders: {e}")
        return []
    
    async def analyze_token_metrics(self, token_address: str) -> Dict[str, Any]:
        """Analyze token metrics for decision making"""
        metrics = {
            "holder_concentration": 0,
            "liquidity_ratio": 0,
            "volume_24h": 0,
            "price_change_24h": 0,
            "whale_activity_score": 0
        }
        
        try:
            holders = await self.get_token_holders(token_address)
            
            if holders:
                # Calculate holder concentration (top 10 holders percentage)
                total_supply = sum(h['amount'] for h in holders)
                top_10_amount = sum(h['amount'] for h in holders[:10])
                metrics["holder_concentration"] = (top_10_amount / total_supply) * 100
                
                # Count whales (holders with >1% of supply)
                whale_count = sum(1 for h in holders if (h['amount'] / total_supply) > 0.01)
                metrics["whale_activity_score"] = min(whale_count * 10, 100)
                
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
            
        return metrics
    
    async def start(self):
        """Start monitoring"""
        logger.info("Starting PumpFun monitor...")
        await self.connect_websocket()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Stopping PumpFun monitor...")
```

### 3.2 DEX Monitor (scrapers/dex.py)

```python
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger

class DEXMonitor:
    """Monitors DEX activity and payments"""
    
    def __init__(self):
        self.raydium_api = "https://api.raydium.io/v2"
        self.orca_api = "https://api.orca.so/v1"
        self.jupiter_api = "https://price.jup.ag/v4"
        self.tracked_tokens = {}
        
    async def check_dex_payment(self, token_address: str) -> Dict[str, Any]:
        """Check if DEX has been paid for a token"""
        dex_info = {
            "is_paid": False,
            "payment_time": None,
            "liquidity_amount": 0,
            "dex_name": None,
            "pool_address": None
        }
        
        # Check Raydium
        raydium_pool = await self.check_raydium_pool(token_address)
        if raydium_pool:
            dex_info["is_paid"] = True
            dex_info["dex_name"] = "Raydium"
            dex_info["pool_address"] = raydium_pool.get("pool_address")
            dex_info["liquidity_amount"] = raydium_pool.get("liquidity", 0)
            dex_info["payment_time"] = raydium_pool.get("created_at")
            
        return dex_info
    
    async def check_raydium_pool(self, token_address: str) -> Optional[Dict]:
        """Check Raydium for pool information"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.raydium_api}/pool/info",
                    params={"mint": token_address}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("success") and data.get("data"):
                            return data["data"][0] if data["data"] else None
        except Exception as e:
            logger.error(f"Raydium API error: {e}")
        return None
    
    async def monitor_post_dex_activity(
        self, 
        token_address: str, 
        duration_hours: int = 1
    ) -> Dict[str, Any]:
        """Monitor whale activity after DEX payment"""
        
        analysis = {
            "whale_sells": 0,
            "whale_buys": 0,
            "net_flow": 0,
            "largest_sell": 0,
            "largest_buy": 0,
            "risk_score": 0,
            "recommendation": "OBSERVE"
        }
        
        # Track token for specified duration
        if token_address not in self.tracked_tokens:
            self.tracked_tokens[token_address] = {
                "start_time": datetime.utcnow(),
                "transactions": []
            }
        
        token_data = self.tracked_tokens[token_address]
        cutoff_time = datetime.utcnow() - timedelta(hours=duration_hours)
        
        # Filter recent transactions
        recent_txs = [
            tx for tx in token_data["transactions"] 
            if tx["timestamp"] > cutoff_time
        ]
        
        for tx in recent_txs:
            if tx["type"] == "SELL" and tx["amount"] > 10:  # 10 SOL threshold
                analysis["whale_sells"] += tx["amount"]
                analysis["largest_sell"] = max(analysis["largest_sell"], tx["amount"])
            elif tx["type"] == "BUY" and tx["amount"] > 10:
                analysis["whale_buys"] += tx["amount"]
                analysis["largest_buy"] = max(analysis["largest_buy"], tx["amount"])
        
        analysis["net_flow"] = analysis["whale_buys"] - analysis["whale_sells"]
        
        # Calculate risk score (0-100)
        if analysis["whale_sells"] > 0:
            sell_ratio = analysis["whale_sells"] / max(analysis["whale_buys"], 1)
            analysis["risk_score"] = min(int(sell_ratio * 50), 100)
        
        # Generate recommendation
        if analysis["risk_score"] > 70:
            analysis["recommendation"] = "AVOID"
        elif analysis["risk_score"] > 40:
            analysis["recommendation"] = "CAUTION"
        elif analysis["net_flow"] > 100:
            analysis["recommendation"] = "POTENTIAL_BUY"
        
        return analysis
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price from Jupiter aggregator"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.jupiter_api}/price",
                    params={"ids": token_address}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if token_address in data.get("data", {}):
                            return data["data"][token_address]["price"]
        except Exception as e:
            logger.error(f"Jupiter price API error: {e}")
        return None
```

## Phase 4: AI Analysis Engine

### 4.1 AI Analyzer (analyzers/ai_analyzer.py)

```python
import openai
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from config import config
from loguru import logger

@dataclass
class MarketAnalysis:
    sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0-1
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    action: str  # 'BUY', 'SELL', 'HOLD', 'OBSERVE'
    reasoning: str
    key_factors: list

class AIAnalyzer:
    """AI-powered market analysis"""
    
    def __init__(self):
        if config.ENABLE_AI_ANALYSIS and config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
            self.ai_enabled = True
        else:
            self.ai_enabled = False
            logger.warning("AI analysis disabled - using rule-based analysis")
    
    async def analyze_market_conditions(
        self, 
        market_data: Dict[str, Any]
    ) -> MarketAnalysis:
        """Analyze market conditions using AI or fallback rules"""
        
        if self.ai_enabled:
            return await self._ai_analysis(market_data)
        else:
            return self._rule_based_analysis(market_data)
    
    async def _ai_analysis(self, market_data: Dict[str, Any]) -> MarketAnalysis:
        """Use OpenAI for market analysis"""
        try:
            prompt = self._build_analysis_prompt(market_data)
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a crypto trading analyst. Provide JSON responses only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return MarketAnalysis(
                sentiment=result.get("sentiment", "NEUTRAL"),
                confidence=result.get("confidence", 0.5),
                risk_level=result.get("risk_level", "MEDIUM"),
                action=result.get("action", "OBSERVE"),
                reasoning=result.get("reasoning", ""),
                key_factors=result.get("key_factors", [])
            )
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._rule_based_analysis(market_data)
    
    def _build_analysis_prompt(self, market_data: Dict[str, Any]) -> str:
        """Build prompt for AI analysis"""
        return f"""
        Analyze this cryptocurrency market data and provide trading recommendations.
        
        Market Data:
        - Token: {market_data.get('token_address', 'Unknown')}
        - Current Price: ${market_data.get('price', 0):.6f}
        - 24h Volume: ${market_data.get('volume_24h', 0):,.2f}
        - 24h Change: {market_data.get('price_change_24h', 0):.2f}%
        - Market Cap: ${market_data.get('market_cap', 0):,.2f}
        - Holder Count: {market_data.get('holder_count', 0)}
        - Liquidity: ${market_data.get('liquidity', 0):,.2f}
        
        Whale Activity:
        - Recent Whale Buys: {market_data.get('whale_buys', 0)} SOL
        - Recent Whale Sells: {market_data.get('whale_sells', 0)} SOL
        - Largest Transaction: {market_data.get('largest_tx', 0)} SOL
        
        Technical Indicators:
        - RSI: {market_data.get('rsi', 50)}
        - MACD Signal: {market_data.get('macd_signal', 'NEUTRAL')}
        - Support Level: ${market_data.get('support', 0):.6f}
        - Resistance Level: ${market_data.get('resistance', 0):.6f}
        
        Respond with JSON only:
        {{
            "sentiment": "BULLISH|BEARISH|NEUTRAL",
            "confidence": 0.0-1.0,
            "risk_level": "LOW|MEDIUM|HIGH",
            "action": "BUY|SELL|HOLD|OBSERVE",
            "reasoning": "Brief explanation",
            "key_factors": ["factor1", "factor2", "factor3"]
        }}
        """
    
    def _rule_based_analysis(self, market_data: Dict[str, Any]) -> MarketAnalysis:
        """Fallback rule-based analysis"""
        sentiment = "NEUTRAL"
        confidence = 0.5
        risk_level = "MEDIUM"
        action = "OBSERVE"
        key_factors = []
        
        # Price change analysis
        price_change = market_data.get('price_change_24h', 0)
        if price_change > 20:
            sentiment = "BULLISH"
            key_factors.append("Strong price momentum")
        elif price_change < -20:
            sentiment = "BEARISH"
            key_factors.append("Significant price decline")
        
        # Whale activity analysis
        whale_buys = market_data.get('whale_buys', 0)
        whale_sells = market_data.get('whale_sells', 0)
        
        if whale_buys > whale_sells * 2:
            sentiment = "BULLISH"
            confidence += 0.2
            key_factors.append("Strong whale accumulation")
        elif whale_sells > whale_buys * 2:
            sentiment = "BEARISH"
            risk_level = "HIGH"
            key_factors.append("Whale distribution detected")
        
        # RSI analysis
        rsi = market_data.get('rsi', 50)
        if rsi < 30:
            action = "BUY" if sentiment != "BEARISH" else "OBSERVE"
            key_factors.append("Oversold condition")
        elif rsi > 70:
            action = "SELL" if sentiment != "BULLISH" else "HOLD"
            key_factors.append("Overbought condition")
        
        # Liquidity check
        liquidity = market_data.get('liquidity', 0)
        if liquidity < 10000:  # Less than $10k liquidity
            risk_level = "HIGH"
            action = "OBSERVE"
            key_factors.append("Low liquidity warning")
        
        confidence = min(max(confidence, 0.1), 0.9)
        
        reasoning = f"Based on {', '.join(key_factors[:2])} with {sentiment.lower()} market conditions."
        
        return MarketAnalysis(
            sentiment=sentiment,
            confidence=confidence,
            risk_level=risk_level,
            action=action,
            reasoning=reasoning,
            key_factors=key_factors
        )
```

## Phase 5: Trading Strategies

### 5.1 Strategy Manager (analyzers/strategies.py)

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import ta  # Technical Analysis library
from loguru import logger

class Strategy(ABC):
    """Base strategy class"""
    
    @abstractmethod
    async def analyze(self, market_data: Dict) -> Dict[str, Any]:
        """Analyze market and return trading signal"""
        pass

class MovingAverageCrossover(Strategy):
    """Simple MA crossover strategy"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 15):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    async def analyze(self, market_data: Dict) -> Dict[str, Any]:
        prices = market_data.get('price_history', [])
        
        if len(prices) < self.slow_period:
            return {"action": "HOLD", "reason": "Insufficient data"}
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['sma_fast'] = df['price'].rolling(window=self.fast_period).mean()
        df['sma_slow'] = df['price'].rolling(window=self.slow_period).mean()
        
        current_fast = df['sma_fast'].iloc[-1]
        current_slow = df['sma_slow'].iloc[-1]
        prev_fast = df['sma_fast'].iloc[-2]
        prev_slow = df['sma_slow'].iloc[-2]
        
        # Bullish crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            return {
                "action": "BUY",
                "reason": f"Bullish MA crossover: {self.fast_period} > {self.slow_period}",
                "confidence": 0.7
            }
        # Bearish crossover
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return {
                "action": "SELL",
                "reason": f"Bearish MA crossover: {self.fast_period} < {self.slow_period}",
                "confidence": 0.7
            }
        
        return {"action": "HOLD", "reason": "No crossover signal"}

class RSIMomentum(Strategy):
    """RSI-based momentum strategy"""
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    async def analyze(self, market_data: Dict) -> Dict[str, Any]:
        prices = market_data.get('price_history', [])
        
        if len(prices) < self.period + 1:
            return {"action": "HOLD", "reason": "Insufficient data"}
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=self.period).rsi()
        
        current_rsi = df['rsi'].iloc[-1]
        
        if current_rsi < self.oversold:
            return {
                "action": "BUY",
                "reason": f"RSI oversold: {current_rsi:.2f}",
                "confidence": 0.6
            }
        elif current_rsi > self.overbought:
            return {
                "action": "SELL",
                "reason": f"RSI overbought: {current_rsi:.2f}",
                "confidence": 0.6
            }
        
        return {
            "action": "HOLD",
            "reason": f"RSI neutral: {current_rsi:.2f}"
        }

class VolumeProfile(Strategy):
    """Volume-based trading strategy"""
    
    async def analyze(self, market_data: Dict) -> Dict[str, Any]:
        volume_24h = market_data.get('volume_24h', 0)
        avg_volume = market_data.get('avg_volume_7d', 0)
        price_change = market_data.get('price_change_1h', 0)
        
        if avg_volume == 0:
            return {"action": "HOLD", "reason": "No volume data"}
        
        volume_ratio = volume_24h / avg_volume
        
        # High volume + price increase = Strong buy signal
        if volume_ratio > 2 and price_change > 5:
            return {
                "action": "BUY",
                "reason": f"Volume surge {volume_ratio:.1f}x with positive momentum",
                "confidence": 0.8
            }
        # High volume + price decrease = Potential reversal or further decline
        elif volume_ratio > 2 and price_change < -5:
            return {
                "action": "SELL",
                "reason": f"Volume surge {volume_ratio:.1f}x with negative momentum",
                "confidence": 0.7
            }
        
        return {"action": "HOLD", "reason": "Normal volume levels"}

class WhaleFollower(Strategy):
    """Follow whale trading patterns"""
    
    async def analyze(self, market_data: Dict) -> Dict[str, Any]:
        whale_buys = market_data.get('whale_buys_1h', 0)
        whale_sells = market_data.get('whale_sells_1h', 0)
        
        net_whale_flow = whale_buys - whale_sells
        
        if net_whale_flow > 50:  # Net 50 SOL whale buying
            return {
                "action": "BUY",
                "reason": f"Whales accumulating: +{net_whale_flow:.1f} SOL",
                "confidence": 0.75
            }
        elif net_whale_flow < -50:  # Net 50 SOL whale selling
            return {
                "action": "SELL",
                "reason": f"Whales distributing: {net_whale_flow:.1f} SOL",
                "confidence": 0.75
            }
        
        return {
            "action": "HOLD",
            "reason": f"Neutral whale activity: {net_whale_flow:.1f} SOL"
        }

class StrategyManager:
    """Manages multiple strategies and combines signals"""
    
    def __init__(self):
        self.strategies = {
            "ma_crossover": MovingAverageCrossover(),
            "rsi_momentum": RSIMomentum(),
            "volume_profile": VolumeProfile(),
            "whale_follower": WhaleFollower()
        }
        self.weights = {
            "ma_crossover": 0.2,
            "rsi_momentum": 0.2,
            "volume_profile": 0.25,
            "whale_follower": 0.35
        }
    
    async def get_combined_signal(self, market_data: Dict) -> Dict[str, Any]:
        """Combine signals from multiple strategies"""
        signals = {}
        total_confidence = 0
        buy_score = 0
        sell_score = 0
        
        for name, strategy in self.strategies.items():
            try:
                signal = await strategy.analyze(market_data)
                signals[name] = signal
                
                weight = self.weights.get(name, 0.25)
                confidence = signal.get('confidence', 0.5)
                
                if signal['action'] == 'BUY':
                    buy_score += weight * confidence
                elif signal['action'] == 'SELL':
                    sell_score += weight * confidence
                    
                total_confidence += confidence * weight
                
            except Exception as e:
                logger.error(f"Strategy {name} failed: {e}")
                signals[name] = {"action": "HOLD", "reason": "Error"}
        
        # Determine final action
        if buy_score > 0.5:
            action = "BUY"
            confidence = buy_score
        elif sell_score > 0.5:
            action = "SELL"
            confidence = sell_score
        else:
            action = "HOLD"
            confidence = 0.5
        
        return {
            "action": action,
            "confidence": confidence,
            "signals": signals,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "timestamp": datetime.utcnow().isoformat()
        }
```

## Phase 6: Trading Engine

### 6.1 Demo Trader (traders/demo_trader.py)

```python
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
from loguru import logger

@dataclass
class Position:
    token_address: str
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0
    pnl: float = 0
    pnl_percent: float = 0

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_history: List[Dict] = field(default_factory=list)
    total_value: float = 0
    total_pnl: float = 0
    win_rate: float = 0
    total_trades: int = 0
    winning_trades: int = 0

class DemoTrader:
    """Paper trading implementation"""
    
    def __init__(self, initial_balance: float = 1000):
        self.portfolio = Portfolio(cash=initial_balance)
        self.initial_balance = initial_balance
        self.max_position_size = 0.1  # 10% max per position
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit
    
    async def execute_trade(
        self, 
        signal: Dict[str, Any], 
        current_price: float,
        token_address: str
    ) -> Dict[str, Any]:
        """Execute a demo trade based on signal"""
        
        trade_result = {
            "success": False,
            "action": signal['action'],
            "token": token_address,
            "price": current_price,
            "timestamp": datetime.utcnow().isoformat(),
            "reason": signal.get('reason', ''),
            "error": None
        }
        
        try:
            if signal['action'] == 'BUY':
                result = await self._execute_buy(token_address, current_price, signal)
            elif signal['action'] == 'SELL':
                result = await self._execute_sell(token_address, current_price, signal)
            else:
                result = {"message": "HOLD - No action taken"}
            
            trade_result.update(result)
            trade_result["success"] = True
            
        except Exception as e:
            trade_result["error"] = str(e)
            logger.error(f"Trade execution error: {e}")
        
        # Update portfolio metrics
        await self._update_portfolio_metrics()
        
        return trade_result
    
    async def _execute_buy(
        self, 
        token_address: str, 
        price: float, 
        signal: Dict
    ) -> Dict[str, Any]:
        """Execute buy order"""
        
        # Check if already have position
        if token_address in self.portfolio.positions:
            return {"message": "Position already exists", "action_taken": False}
        
        # Calculate position size
        position_value = self.portfolio.cash * self.max_position_size
        amount = position_value / price
        
        # Check available cash
        if position_value > self.portfolio.cash:
            return {"message": "Insufficient funds", "action_taken": False}
        
        # Create position
        position = Position(
            token_address=token_address,
            amount=amount,
            entry_price=price,
            entry_time=datetime.utcnow(),
            current_price=price
        )
        
        # Update portfolio
        self.portfolio.cash -= position_value
        self.portfolio.positions[token_address] = position
        
        # Record trade
        trade = {
            "type": "BUY",
            "token": token_address,
            "amount": amount,
            "price": price,
            "value": position_value,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": signal
        }
        self.portfolio.trade_history.append(trade)
        self.portfolio.total_trades += 1
        
        logger.info(f"Demo BUY executed: {amount:.4f} @ ${price:.6f}")
        
        return {
            "action_taken": True,
            "amount": amount,
            "value": position_value,
            "remaining_cash": self.portfolio.cash
        }
    
    async def _execute_sell(
        self, 
        token_address: str, 
        price: float, 
        signal: Dict
    ) -> Dict[str, Any]:
        """Execute sell order"""
        
        # Check if position exists
        if token_address not in self.portfolio.positions:
            return {"message": "No position to sell", "action_taken": False}
        
        position = self.portfolio.positions[token_address]
        sell_value = position.amount * price
        pnl = sell_value - (position.amount * position.entry_price)
        pnl_percent = (pnl / (position.amount * position.entry_price)) * 100
        
        # Update portfolio
        self.portfolio.cash += sell_value
        del self.portfolio.positions[token_address]
        
        # Update metrics
        self.portfolio.total_pnl += pnl
        if pnl > 0:
            self.portfolio.winning_trades += 1
        
        # Record trade
        trade = {
            "type": "SELL",
            "token": token_address,
            "amount": position.amount,
            "price": price,
            "value": sell_value,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": signal
        }
        self.portfolio.trade_history.append(trade)
        
        logger.info(f"Demo SELL executed: {position.amount:.4f} @ ${price:.6f} | PnL: ${pnl:.2f} ({pnl_percent:.2f}%)")
        
        return {
            "action_taken": True,
            "amount": position.amount,
            "value": sell_value,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "remaining_cash": self.portfolio.cash
        }
    
    async def check_stop_loss_take_profit(self, current_prices: Dict[str, float]):
        """Check and execute stop loss or take profit"""
        
        for token_address, position in list(self.portfolio.positions.items()):
            if token_address not in current_prices:
                continue
            
            current_price = current_prices[token_address]
            position.current_price = current_price
            
            # Calculate PnL
            pnl_percent = ((current_price - position.entry_price) / position.entry_price) * 100
            
            # Check stop loss
            if pnl_percent <= -self.stop_loss * 100:
                logger.warning(f"Stop loss triggered for {token_address}")
                await self._execute_sell(
                    token_address, 
                    current_price,
                    {"action": "SELL", "reason": "Stop loss triggered"}
                )
            
            # Check take profit
            elif pnl_percent >= self.take_profit * 100:
                logger.info(f"Take profit triggered for {token_address}")
                await self._execute_sell(
                    token_address,
                    current_price,
                    {"action": "SELL", "reason": "Take profit triggered"}
                )
    
    async def _update_portfolio_metrics(self):
        """Update portfolio metrics"""
        
        # Calculate total value
        positions_value = sum(
            pos.amount * pos.current_price 
            for pos in self.portfolio.positions.values()
        )
        self.portfolio.total_value = self.portfolio.cash + positions_value
        
        # Calculate win rate
        if self.portfolio.total_trades > 0:
            self.portfolio.win_rate = (
                self.portfolio.winning_trades / self.portfolio.total_trades
            ) * 100
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        
        return {
            "cash": self.portfolio.cash,
            "positions": [
                {
                    "token": pos.token_address,
                    "amount": pos.amount,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "pnl": (pos.current_price - pos.entry_price) * pos.amount,
                    "pnl_percent": ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                }
                for pos in self.portfolio.positions.values()
            ],
            "total_value": self.portfolio.total_value,
            "total_pnl": self.portfolio.total_pnl,
            "total_pnl_percent": ((self.portfolio.total_value - self.initial_balance) / self.initial_balance) * 100,
            "win_rate": self.portfolio.win_rate,
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades
        }
    
    def export_trade_history(self, filepath: str):
        """Export trade history to JSON file"""
        
        with open(filepath, 'w') as f:
            json.dump({
                "portfolio_status": self.get_portfolio_status(),
                "trade_history": self.portfolio.trade_history
            }, f, indent=2)
        
        logger.info(f"Trade history exported to {filepath}")
```

## Phase 7: Telegram Bot

### 7.1 Telegram Notifications (notifications/telegram_bot.py)

```python
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from typing import Dict, Any
from datetime import datetime
from config import config
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
        
        # Start bot
        await self.app.initialize()
        await self.app.start()
        logger.info("Telegram bot initialized")
    
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
```

## Phase 8: Main Orchestrator

### 8.1 Main Application (main.py)

```python
import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

# Import all components
from config import config
from database.manager import db
from scrapers.pumpfun import PumpFunMonitor
from scrapers.dex import DEXMonitor
from scrapers.market import MarketDataCollector
from analyzers.ai_analyzer import AIAnalyzer
from analyzers.strategies import StrategyManager
from traders.demo_trader import DemoTrader
from traders.live_trader import LiveTrader
from notifications.telegram_bot import TelegramBot

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
                
                if post_dex_analysis['risk_score'] < 50:
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
                strategy_signal['confidence'] > 0.6):
                
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
                
                if signal['action'] == 'SELL' and signal['confidence'] > 0.6:
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
```

## Phase 9: Docker Deployment

### 9.1 Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: cryptobot
      POSTGRES_USER: cryptouser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  bot:
    build: .
    environment:
      - DATABASE_URL=postgresql://cryptouser:${DB_PASSWORD}@postgres/cryptobot
      - REDIS_URL=redis://redis:6379
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - SOLANA_RPC_URL=${SOLANA_RPC_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEMO_MODE=${DEMO_MODE}
    volumes:
      - ./logs:/app/logs
      - ./reports:/app/reports
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

volumes:
  postgres_data:
```

### 9.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config.py .
COPY main.py .

# Create directories
RUN mkdir -p logs reports

# Run bot
CMD ["python", "main.py"]
```

## Installation & Running Instructions

### Step 1: Clone and Setup
```bash
# Create project directory
mkdir crypto-trading-bot
cd crypto-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

### Step 3: Database Setup
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
python -c "from database.manager import db; db.initialize()"
```

### Step 4: Run the Bot
```bash
# Development mode
python main.py

# Production with Docker
docker-compose up -d
```

### Step 5: Monitor
```bash
# View logs
tail -f logs/bot_*.log

# Check Docker logs
docker-compose logs -f bot
```

## Safety Reminders

1. **ALWAYS start in DEMO mode** for at least 30 days
2. **Never risk more than you can afford to lose**
3. **Monitor the bot regularly** - it's not set-and-forget
4. **Keep your API keys secure** - never commit them to git
5. **Test thoroughly** before going live
6. **Stay paranoid** about market anomalies and suspicious activity

This Python implementation gives you complete control without relying on N8N, with all the same features plus additional flexibility for customization!
