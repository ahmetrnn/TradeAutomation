import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from ..config import config
from loguru import logger

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available")

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
        if config.ENABLE_AI_ANALYSIS and config.OPENAI_API_KEY and OPENAI_AVAILABLE:
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
        price_change_1h = market_data.get('price_change_1h', 0)

        # More aggressive thresholds for testing
        if price_change > 5 or price_change_1h > 2:
            sentiment = "BULLISH"
            key_factors.append("Positive price momentum")
            action = "BUY"
            confidence = 0.6
        elif price_change < -5 or price_change_1h < -2:
            sentiment = "BEARISH"
            key_factors.append("Negative price movement")
            action = "SELL"

        # Whale activity analysis (more sensitive for testing)
        whale_buys = market_data.get('whale_buys_1h', 0)
        whale_sells = market_data.get('whale_sells_1h', 0)

        if whale_buys > whale_sells * 1.2:
            sentiment = "BULLISH"
            confidence += 0.2
            action = "BUY"
            key_factors.append("Whale accumulation detected")
        elif whale_sells > whale_buys * 1.2:
            sentiment = "BEARISH"
            risk_level = "MEDIUM"
            action = "SELL"
            key_factors.append("Whale distribution activity")

        # RSI analysis
        rsi = market_data.get('rsi', 50)
        if rsi < 30:
            action = "BUY" if sentiment != "BEARISH" else "OBSERVE"
            key_factors.append("Oversold condition")
        elif rsi > 70:
            action = "SELL" if sentiment != "BULLISH" else "HOLD"
            key_factors.append("Overbought condition")

        # Liquidity check (relaxed for testing)
        liquidity = market_data.get('liquidity', 0)
        if liquidity < 1000:  # Less than $1k liquidity
            risk_level = "HIGH"
            key_factors.append("Very low liquidity warning")

        # Default to BUY for testing if no strong signals
        if action == "OBSERVE" and len(key_factors) > 0:
            action = "BUY"
            confidence = 0.5
            logger.info("Testing mode: defaulting to BUY action")

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
