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
        self.max_open_positions = 5  # Max 5 positions at once
        self.stop_loss = 0.05  # 5% stop loss
        self.take_profit = 0.15  # 15% take profit
        self.daily_loss = 0  # Track daily losses
        self.daily_loss_limit = 100  # Max $100 loss per day
        self.max_drawdown = 0.20  # 20% max drawdown

    async def execute_trade(
        self,
        signal: Dict[str, Any],
        current_price: float,
        token_address: str
    ) -> Dict[str, Any]:
        """Execute a demo trade based on signal"""

        # Risk management checks
        if not self._check_risk_limits():
            logger.warning("Risk limits exceeded - trade blocked")
            return {
                "success": False,
                "action": signal['action'],
                "token": token_address,
                "price": current_price,
                "reason": "Risk limits exceeded",
                "message": "Daily loss limit or max drawdown reached"
            }

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

        # Check position limits
        if len(self.portfolio.positions) >= self.max_open_positions:
            logger.warning(f"Max open positions ({self.max_open_positions}) reached - skipping buy")
            return {"message": f"Max open positions ({self.max_open_positions}) reached", "action_taken": False}

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

    def _check_risk_limits(self) -> bool:
        """Check if risk management limits allow trading"""

        # Check daily loss limit
        if self.daily_loss >= self.daily_loss_limit:
            logger.error(f"Daily loss limit reached: ${self.daily_loss:.2f} >= ${self.daily_loss_limit}")
            return False

        # Check max drawdown
        if self.portfolio.total_value > 0:
            drawdown = (self.initial_balance - self.portfolio.total_value) / self.initial_balance
            if drawdown >= self.max_drawdown:
                logger.error(f"Max drawdown reached: {drawdown:.1%} >= {self.max_drawdown:.1%}")
                return False

        return True

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
