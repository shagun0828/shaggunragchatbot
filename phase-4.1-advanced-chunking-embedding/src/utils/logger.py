"""
Enhanced logging utility for Phase 4.1
Structured logging with performance tracking
"""

import logging
import sys
import time
from typing import Optional, Dict, Any
import structlog
from pythonjsonlogger import jsonlogger


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a structured logger for Phase 4.1"""
    
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
    
    def log_chunking_metrics(self, fund_name: str, chunk_count: int, 
                           processing_time: float, **metadata) -> None:
        """Log chunking-specific metrics"""
        self.logger.info(
            "Chunking metrics",
            fund_name=fund_name,
            chunk_count=chunk_count,
            processing_time=processing_time,
            chunks_per_second=chunk_count / processing_time if processing_time > 0 else 0,
            **metadata
        )
    
    def log_embedding_metrics(self, chunk_count: int, embedding_time: float, 
                            quality_score: float, **metadata) -> None:
        """Log embedding-specific metrics"""
        self.logger.info(
            "Embedding metrics",
            chunk_count=chunk_count,
            embedding_time=embedding_time,
            quality_score=quality_score,
            embeddings_per_second=chunk_count / embedding_time if embedding_time > 0 else 0,
            **metadata
        )
    
    def log_quality_metrics(self, quality_report: Dict[str, Any], **metadata) -> None:
        """Log quality-specific metrics"""
        self.logger.info(
            "Quality metrics",
            overall_score=quality_report.get('overall_score', 0),
            total_issues=sum(len(indices) for indices in quality_report.get('issues', {}).values()),
            duplicate_rate=quality_report.get('statistics', {}).get('duplicate_rate', 0),
            outlier_rate=quality_report.get('statistics', {}).get('outlier_rate', 0),
            **metadata
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
