from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from contextlib import contextmanager
from .models import Base
from ..config import config
import logging
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.enabled = os.getenv('SKIP_DATABASE', 'false').lower() != 'true'

        if not self.enabled:
            logger.warning("Database disabled - running without persistence")
            self.engine = None
            self.SessionLocal = None
            return

        try:
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
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.warning("Running without database")
            self.enabled = False
            self.engine = None
            self.SessionLocal = None

    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        if not self.enabled:
            logger.debug("Database disabled - session not created")
            yield None
            return

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
        if not self.enabled:
            logger.debug("Database disabled - whale alert not saved")
            return None

        try:
            with self.get_session() as session:
                if session is None:
                    return None
                from .models import WhaleAlert
                alert = WhaleAlert(**alert_data)
                session.add(alert)
                return alert.id
        except Exception as e:
            logger.error(f"Failed to save whale alert: {e}")
            return None

    def get_recent_whale_alerts(self, hours=24):
        if not self.enabled:
            logger.debug("Database disabled - returning empty alerts")
            return []

        try:
            with self.get_session() as session:
                if session is None:
                    return []
                from .models import WhaleAlert
                from datetime import datetime, timedelta

                cutoff = datetime.utcnow() - timedelta(hours=hours)
                alerts = session.query(WhaleAlert)\
                    .filter(WhaleAlert.timestamp > cutoff)\
                    .order_by(WhaleAlert.timestamp.desc())\
                    .all()
                return [alert.to_dict() for alert in alerts]
        except Exception as e:
            logger.error(f"Failed to get whale alerts: {e}")
            return []

db = DatabaseManager()
