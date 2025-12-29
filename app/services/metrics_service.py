"""Prometheus metrics service for LLM Gateway."""

import logging
from typing import Optional

from flask import Flask, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry
)

from app.config.settings import get_gateway_settings


logger = logging.getLogger(__name__)


class MetricsService:
    """Service for collecting and exposing Prometheus metrics."""
    
    def __init__(self):
        """Initialize metrics collectors."""
        self.gateway_settings = get_gateway_settings()
        self.registry = CollectorRegistry()
        
        # Request counter
        self.request_counter = Counter(
            "llm_gateway_requests_total",
            "Total number of LLM requests",
            ["model", "status", "cached"],
            registry=self.registry
        )
        
        # Token counters
        self.token_counter = Counter(
            "llm_gateway_tokens_total",
            "Total number of tokens processed",
            ["model", "type"],
            registry=self.registry
        )
        
        # Cost counter
        self.cost_counter = Counter(
            "llm_gateway_cost_usd_total",
            "Total cost in USD",
            ["model"],
            registry=self.registry
        )
        
        # Latency histogram
        self.latency_histogram = Histogram(
            "llm_gateway_request_latency_seconds",
            "Request latency in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry
        )
        
        # Active requests gauge
        self.active_requests_gauge = Gauge(
            "llm_gateway_active_requests",
            "Number of currently active requests",
            ["model"],
            registry=self.registry
        )
        
        # Model availability gauge
        self.model_available_gauge = Gauge(
            "llm_gateway_model_available",
            "Whether a model is available (1) or not (0)",
            ["model"],
            registry=self.registry
        )
    
    def record_request(
        self,
        model_name: str,
        status: str,
        cached: bool,
        latency_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: Optional[float]
    ):
        """Record metrics for a completed request."""
        if not self.gateway_settings.metrics_enabled:
            return
        
        # Increment request counter
        self.request_counter.labels(
            model=model_name,
            status=status,
            cached=str(cached).lower()
        ).inc()
        
        # Record tokens
        self.token_counter.labels(model=model_name, type="prompt").inc(prompt_tokens)
        self.token_counter.labels(model=model_name, type="completion").inc(completion_tokens)
        
        # Record cost
        if cost_usd is not None:
            self.cost_counter.labels(model=model_name).inc(cost_usd)
        
        # Record latency
        self.latency_histogram.labels(model=model_name).observe(latency_seconds)
    
    def increment_active_requests(self, model_name: str):
        """Increment active request counter."""
        if self.gateway_settings.metrics_enabled:
            self.active_requests_gauge.labels(model=model_name).inc()
    
    def decrement_active_requests(self, model_name: str):
        """Decrement active request counter."""
        if self.gateway_settings.metrics_enabled:
            self.active_requests_gauge.labels(model=model_name).dec()
    
    def set_model_availability(self, model_name: str, available: bool):
        """Set model availability status."""
        if self.gateway_settings.metrics_enabled:
            self.model_available_gauge.labels(model=model_name).set(1 if available else 0)
    
    def get_metrics_response(self) -> Response:
        """Generate Prometheus metrics response."""
        return Response(
            generate_latest(self.registry),
            mimetype=CONTENT_TYPE_LATEST
        )


# Global metrics instance
metrics_service = MetricsService()

