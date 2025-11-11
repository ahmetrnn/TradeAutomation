from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

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
    meta_data = Column(JSON, default=dict)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'token_address': self.token_address,
            'whale_address': self.whale_address,
            'amount_sol': self.amount_sol,
            'tx_hash': self.transaction_hash,
            'meta_data': self.meta_data
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
    meta_data = Column(JSON, default=dict)

class SystemLog(Base):
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(20))  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(50))
    message = Column(String(500))
    details = Column(JSON, default=dict)
