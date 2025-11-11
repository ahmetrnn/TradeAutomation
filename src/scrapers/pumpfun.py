import asyncio
import json
import websockets
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from loguru import logger
from ..config import config

class PumpFunMonitor:
    """Monitors PumpFun for whale transactions"""

    def __init__(self, on_whale_detected: Optional[Callable] = None):
        self.ws_url = config.PUMPFUN_WS_URL
        self.solana_client = AsyncClient(config.SOLANA_RPC_URL)
        self.on_whale_detected = on_whale_detected
        self.whale_threshold = config.WHALE_THRESHOLD_SOL
        self.running = False
        self.retry_count = 0
        self.max_retries = 999  # Nearly unlimited retries for production
        self.base_delay = 5  # Start with 5 seconds

    async def connect_websocket(self):
        """Establish WebSocket connection using native Solana RPC for PumpFun monitoring"""
        try:
            # Convert HTTP RPC URL to WebSocket
            ws_rpc_url = config.SOLANA_RPC_URL.replace('https://', 'wss://').replace('http://', 'ws://')

            # Pump.fun program ID on Solana
            PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

            async with websockets.connect(ws_rpc_url, ssl=True) as websocket:
                logger.info(f"Connected to Solana WebSocket for Pump.fun monitoring")

                # Subscribe to Pump.fun program logs using logsSubscribe
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {
                            "mentions": [PUMPFUN_PROGRAM_ID]
                        },
                        {
                            "commitment": "confirmed"
                        }
                    ]
                }
                await websocket.send(json.dumps(subscribe_msg))

                # Wait for subscription confirmation
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info(f"Subscription confirmed: {response_data.get('result', 'success')}")

                self.running = True
                while self.running:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=60.0
                        )
                        await self.process_message(message)
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        ping_msg = {"jsonrpc": "2.0", "id": 999, "method": "ping"}
                        await websocket.send(json.dumps(ping_msg))
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except websockets.exceptions.WebSocketException as e:
            logger.warning(f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket: {e}")
        finally:
            # Always attempt reconnection with exponential backoff
            if self.running and self.retry_count < self.max_retries:
                delay = min(self.base_delay * (2 ** min(self.retry_count, 6)), 300)  # Max 5 minutes
                self.retry_count += 1
                logger.info(f"Reconnecting in {delay}s... (attempt {self.retry_count})")
                await asyncio.sleep(delay)
                await self.connect_websocket()
            elif self.retry_count >= self.max_retries:
                logger.error("Max reconnection attempts reached")
                self.running = False

    async def process_message(self, message: str):
        """Process incoming WebSocket message from Solana logs"""
        try:
            data = json.loads(message)

            # Handle Solana logsSubscribe notifications
            if "params" in data:
                result = data["params"].get("result", {})

                if "value" in result:
                    value = result["value"]
                    signature = value.get("signature", "Unknown")
                    logs = value.get("logs", [])

                    # Log Pump.fun activity detected
                    logger.info(f"Pump.fun transaction detected: {signature[:16]}...")

                    # Look for specific Pump.fun events in logs
                    # This is a simplified version - would need proper parsing
                    for log in logs:
                        if "Program log:" in log:
                            logger.debug(f"Log: {log}")

                    # Parse real transaction data from logs
                    if len(logs) > 0:
                        # Try to extract real data from transaction logs
                        sol_amount = None
                        token_address = None
                        instruction_type = None

                        # Parse logs for transaction details
                        for log in logs:
                            # Look for swap events with amounts
                            if "SwapEvent" in log:
                                try:
                                    # Extract amount_in (SOL amount in lamports)
                                    if "amount_in:" in log:
                                        import re
                                        match = re.search(r'amount_in:\s*(\d+)', log)
                                        if match:
                                            lamports = int(match.group(1))
                                            sol_amount = lamports / 1_000_000_000  # Convert to SOL
                                except Exception as e:
                                    logger.debug(f"Failed to parse amount: {e}")

                            # Detect instruction type
                            if "Instruction: Buy" in log:
                                instruction_type = "BUY"
                            elif "Instruction: Sell" in log:
                                instruction_type = "SELL"

                        # For now, use signature as token identifier since we can't decode full transaction
                        # In production, you'd decode the transaction accounts to get actual token mint
                        if sol_amount and sol_amount >= self.whale_threshold:
                            token_address = signature[:16]  # Use partial signature as identifier

                            transaction = {
                                "token_address": token_address,
                                "buyer_address": "unknown",
                                "sol_amount": sol_amount,
                                "signature": signature,
                                "token_name": f"Token_{signature[:8]}",
                                "price": 0.0001,  # Would need to calculate from swap amounts
                                "market_cap": 0,
                                "instruction_type": instruction_type
                            }

                            logger.info(f"Whale activity detected: {transaction['sol_amount']:.4f} SOL ({instruction_type})")
                            await self.handle_whale_transaction(transaction)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def handle_whale_transaction(self, transaction: Dict[str, Any]):
        """Handle detected whale transaction"""
        whale_data = {
            "token_address": transaction.get("token_address"),
            "whale_address": transaction.get("buyer_address"),
            "amount_sol": transaction.get("sol_amount"),
            "transaction_hash": transaction.get("signature"),
            "timestamp": datetime.utcnow(),
            "meta_data": {
                "token_name": transaction.get("token_name", "Unknown"),
                "price_per_token": transaction.get("price", 0),
                "market_cap": transaction.get("market_cap", 0)
            }
        }

        logger.info(f"🐋 Whale detected: {whale_data['amount_sol']} SOL on {whale_data['meta_data']['token_name']}")

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
