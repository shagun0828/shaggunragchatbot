"""
Logger utility for Phase 4.3
Structured logging with performance tracking
"""

import logging
import sys
import time
from typing import Optional, Dict, Any
import structlog
from pythonjsonlogger import jsonlogger


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a structured logger for Phase 4.3"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get logger
    logger = structlog.get_logger(name)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)
    
    return logger


class PerformanceLogger:
    """Performance tracking logger"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = structlog.get_logger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.start_times[operation] = time.time()
        self.logger.info(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, **metadata) -> float:
        """End timing an operation and log duration"""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(
            f"Completed operation: {operation}",
            duration=duration,
            **metadata
        )
        
        return duration
    
    def log_multi_model_metrics(self, model_type: str, url_count: int, 
                               processing_time: float, quality_score: float, **metadata) -> None:
        """Log multi-model specific metrics"""
        self.logger.info(
            f"Multi-model metrics: {model_type}",
            model_type=model_type,
            url_count=url_count,
            processing_time=processing_time,
            quality_score=quality_score,
            throughput=url_count / processing_time if processing_time > 0 else 0,
            **metadata
        )
    
    def log_routing_decision(self, url: str, model_type: str, reasoning: str, 
                           confidence: float) -> None:
        """Log routing decision"""
        self.logger.info(
            f"Routing decision made",
            url=url,
            model_type=model_type,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def log_batch_processing(self, batch_name: str, model_type: str, 
                           batch_size: int, processing_time: float) -> None:
        """Log batch processing metrics"""
        self.logger.info(
            f"Batch processed: {batch_name}",
            batch_name=batch_name,
            model_type=model_type,
            batch_size=batch_size,
            processing_time=processing_time,
            throughput=batch_size / processing_time if processing_time > 0 else 0
        )


def get_console_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a console logger for development"""
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger
