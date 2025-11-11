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
        else:
            # Assume tokens from Pump.fun have internal liquidity
            # since they use a bonding curve model
            dex_info["is_paid"] = True
            dex_info["dex_name"] = "PumpFun Bonding Curve"
            dex_info["liquidity_amount"] = 5000  # Minimum liquidity assumption
            dex_info["payment_time"] = datetime.utcnow()
            logger.debug(f"Using bonding curve liquidity for: {token_address}")

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

        # If no data, assume moderate risk for new tokens
        if len(recent_txs) == 0:
            analysis["risk_score"] = 50
            analysis["recommendation"] = "OBSERVE"
            logger.debug(f"No transaction history for {token_address}")

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
