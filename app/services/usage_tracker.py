"""Usage tracking service for request logging and analytics."""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from app.config.settings import get_gateway_settings


logger = logging.getLogger(__name__)


class UsageTrackerService:
    """Service for tracking and analyzing LLM request usage."""
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize the usage tracker with SQLite database."""
        if database_path is None:
            data_directory = Path(__file__).parent.parent.parent / "data"
            data_directory.mkdir(exist_ok=True)
            database_path = str(data_directory / "usage.db")
        
        self.database_path = database_path
        self._thread_local_storage = threading.local()
        self.gateway_settings = get_gateway_settings()
        
        self._initialize_database_schema()
    
    def _get_database_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._thread_local_storage, "connection"):
            self._thread_local_storage.connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False
            )
            self._thread_local_storage.connection.row_factory = sqlite3.Row
        return self._thread_local_storage.connection
    
    def _initialize_database_schema(self):
        """Create database tables if they don't exist."""
        connection = self._get_database_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                api_key_hash TEXT,
                model_name TEXT NOT NULL,
                provider_model TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL,
                latency_ms INTEGER,
                status_code INTEGER,
                cached INTEGER DEFAULT 0,
                error_message TEXT,
                request_metadata TEXT
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON request_logs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_key_hash 
            ON request_logs(api_key_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_name 
            ON request_logs(model_name)
        """)
        
        connection.commit()
        logger.info(f"Usage database initialized at {self.database_path}")
    
    def log_request(
        self,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: Optional[float] = None,
        latency_ms: int = 0,
        status_code: int = 200,
        cached: bool = False,
        api_key_hash: Optional[str] = None,
        provider_model: Optional[str] = None,
        error_message: Optional[str] = None,
        request_metadata: Optional[dict] = None
    ):
        """Log a request to the database."""
        if not self.gateway_settings.request_logging_enabled:
            return
        
        try:
            connection = self._get_database_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO request_logs (
                    timestamp, api_key_hash, model_name, provider_model,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, latency_ms, status_code, cached,
                    error_message, request_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                api_key_hash,
                model_name,
                provider_model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                latency_ms,
                status_code,
                1 if cached else 0,
                error_message,
                json.dumps(request_metadata) if request_metadata else None
            ))
            
            connection.commit()
        except Exception as logging_error:
            logger.error(f"Failed to log request: {logging_error}")
    
    def get_usage_summary(
        self,
        api_key_hash: Optional[str] = None,
        days: int = 30,
        model_name: Optional[str] = None
    ) -> dict[str, Any]:
        """Get aggregated usage statistics."""
        connection = self._get_database_connection()
        cursor = connection.cursor()
        
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Build query with filters
        query_conditions = ["timestamp >= ?"]
        query_parameters = [cutoff_date]
        
        if api_key_hash:
            query_conditions.append("api_key_hash = ?")
            query_parameters.append(api_key_hash)
        
        if model_name:
            query_conditions.append("model_name = ?")
            query_parameters.append(model_name)
        
        where_clause = " AND ".join(query_conditions)
        
        # Get totals
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_requests,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost_usd) as total_cost_usd,
                AVG(latency_ms) as avg_latency_ms,
                SUM(CASE WHEN cached = 1 THEN 1 ELSE 0 END) as cached_requests,
                SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_requests
            FROM request_logs
            WHERE {where_clause}
        """, query_parameters)
        
        totals_row = cursor.fetchone()
        
        # Get per-model breakdown
        cursor.execute(f"""
            SELECT 
                model_name,
                COUNT(*) as requests,
                SUM(total_tokens) as tokens,
                SUM(cost_usd) as cost_usd,
                AVG(latency_ms) as avg_latency_ms
            FROM request_logs
            WHERE {where_clause}
            GROUP BY model_name
            ORDER BY requests DESC
        """, query_parameters)
        
        model_breakdown = [dict(row) for row in cursor.fetchall()]
        
        return {
            "period_days": days,
            "totals": {
                "requests": totals_row["total_requests"] or 0,
                "prompt_tokens": totals_row["total_prompt_tokens"] or 0,
                "completion_tokens": totals_row["total_completion_tokens"] or 0,
                "total_tokens": totals_row["total_tokens"] or 0,
                "cost_usd": round(totals_row["total_cost_usd"] or 0, 4),
                "avg_latency_ms": round(totals_row["avg_latency_ms"] or 0, 2),
                "cached_requests": totals_row["cached_requests"] or 0,
                "error_requests": totals_row["error_requests"] or 0,
                "cache_hit_rate": round(
                    (totals_row["cached_requests"] or 0) / 
                    max(totals_row["total_requests"] or 1, 1) * 100,
                    2
                )
            },
            "by_model": model_breakdown
        }
    
    def get_recent_requests(
        self,
        limit: int = 50,
        api_key_hash: Optional[str] = None
    ) -> list[dict]:
        """Get recent request logs."""
        connection = self._get_database_connection()
        cursor = connection.cursor()
        
        if api_key_hash:
            cursor.execute("""
                SELECT * FROM request_logs
                WHERE api_key_hash = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (api_key_hash, limit))
        else:
            cursor.execute("""
                SELECT * FROM request_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]

