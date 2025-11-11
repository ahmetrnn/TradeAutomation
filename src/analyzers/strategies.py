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
        if volume_ratio > 1.2 and price_change > 2:
            return {
                "action": "BUY",
                "reason": f"Volume surge {volume_ratio:.1f}x with positive momentum",
                "confidence": 0.8
            }
        # High volume + price decrease = Potential reversal or further decline
        elif volume_ratio > 1.2 and price_change < -2:
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

        if net_whale_flow > 5:  # Net 5 SOL whale buying
            return {
                "action": "BUY",
                "reason": f"Whales accumulating: +{net_whale_flow:.1f} SOL",
                "confidence": 0.75
            }
        elif net_whale_flow < -5:  # Net 5 SOL whale selling
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
        if buy_score > 0.3:
            action = "BUY"
            confidence = buy_score
        elif sell_score > 0.3:
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
