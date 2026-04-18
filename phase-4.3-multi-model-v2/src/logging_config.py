"""
Enhanced Logging Configuration for Phase 4.3 Multi-Model System
Comprehensive logging setup with file rotation and structured logging
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import traceback


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "process": record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", "msecs", 
                          "relativeCreated", "thread", "threadName", "processName", 
                          "process", "getMessage", "exc_info", "exc_text", "stack_info"]:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class PipelineLogger:
    """Enhanced logger for pipeline operations"""
    
    def __init__(self, name: str, log_dir: str = "logs", log_level: str = "INFO"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Text log file
        self.text_log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        # JSON structured log file
        self.json_log_file = self.log_dir / f"{name}_{timestamp}.json"
        
        # Error log file
        self.error_log_file = self.log_dir / f"{name}_errors_{timestamp}.log"
        
        # Performance log file
        self.perf_log_file = self.log_dir / f"{name}_performance_{timestamp}.log"
        
        # Setup logger
        self.logger = self._setup_logger(log_level)
        
        # Performance tracking
        self.performance_data = []
        
        self.logger.info(f"PipelineLogger initialized for {name}")
        self.logger.info(f"Text log: {self.text_log_file}")
        self.logger.info(f"JSON log: {self.json_log_file}")
        self.logger.info(f"Error log: {self.error_log_file}")
        self.logger.info(f"Performance log: {self.perf_log_file}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup comprehensive logger with multiple handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # 1. Console handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 2. Text file handler (DEBUG level)
        text_handler = logging.FileHandler(self.text_log_file, mode='w')
        text_handler.setLevel(logging.DEBUG)
        text_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        text_handler.setFormatter(text_formatter)
        logger.addHandler(text_handler)
        
        # 3. JSON structured handler (DEBUG level)
        json_handler = logging.FileHandler(self.json_log_file, mode='w')
        json_handler.setLevel(logging.DEBUG)
        json_formatter = StructuredFormatter()
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)
        
        # 4. Error-only handler (ERROR level)
        error_handler = logging.FileHandler(self.error_log_file, mode='w')
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s\n'
            'EXCEPTION: %(exc_info)s\n'
            'TRACEBACK: %(exc_text)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
        
        # 5. Performance handler (INFO level)
        perf_handler = logging.FileHandler(self.perf_log_file, mode='w')
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter(
            '%(asctime)s | PERF | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)
        
        return logger
    
    def log_phase_start(self, phase_name: str, metadata: Dict[str, Any] = None):
        """Log phase start with metadata"""
        extra = {
            "event_type": "phase_start",
            "phase": phase_name,
            "metadata": metadata or {}
        }
        self.logger.info(f"Starting phase: {phase_name}", extra=extra)
    
    def log_phase_end(self, phase_name: str, status: str, duration: float, metadata: Dict[str, Any] = None):
        """Log phase completion"""
        extra = {
            "event_type": "phase_end",
            "phase": phase_name,
            "status": status,
            "duration": duration,
            "metadata": metadata or {}
        }
        self.logger.info(f"Phase {phase_name} {status} in {duration:.2f}s", extra=extra)
    
    def log_performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }
        self.performance_data.append(perf_entry)
        
        extra = {
            "event_type": "performance",
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }
        
        # Log to performance file
        perf_logger = logging.getLogger(f"{self.name}_performance")
        perf_handler = logging.FileHandler(self.perf_log_file, mode='a')
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter('%(asctime)s | PERF | %(message)s')
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        perf_logger.info(f"{operation}: {duration:.3f}s")
        perf_logger.removeHandler(perf_handler)
        
        # Also log to main logger
        self.logger.info(f"Performance: {operation} took {duration:.3f}s", extra=extra)
    
    def log_error(self, error: Exception, context: str = "", metadata: Dict[str, Any] = None):
        """Log error with full context"""
        extra = {
            "event_type": "error",
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "metadata": metadata or {}
        }
        self.logger.error(f"Error in {context}: {str(error)}", extra=extra, exc_info=True)
    
    def log_system_info(self):
        """Log system information"""
        import psutil
        import platform
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage('/').total,
            "disk_free": psutil.disk_usage('/').free
        }
        
        extra = {
            "event_type": "system_info",
            "system_info": system_info
        }
        
        self.logger.info("System Information", extra=extra)
    
    def log_environment(self, env_dict: Dict[str, str]):
        """Log environment configuration"""
        extra = {
            "event_type": "environment",
            "environment": env_dict
        }
        self.logger.info("Environment Configuration", extra=extra)
    
    def save_performance_summary(self):
        """Save performance summary to file"""
        if not self.performance_data:
            return
        
        summary_file = self.log_dir / f"{self.name}_performance_summary.json"
        
        # Calculate statistics
        durations = [entry["duration"] for entry in self.performance_data]
        operations = list(set(entry["operation"] for entry in self.performance_data))
        
        operation_stats = {}
        for operation in operations:
            op_data = [entry for entry in self.performance_data if entry["operation"] == operation]
            op_durations = [entry["duration"] for entry in op_data]
            
            operation_stats[operation] = {
                "count": len(op_durations),
                "total_time": sum(op_durations),
                "avg_time": sum(op_durations) / len(op_durations),
                "min_time": min(op_durations),
                "max_time": max(op_durations)
            }
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(self.performance_data),
            "total_time": sum(durations),
            "avg_time": sum(durations) / len(durations),
            "min_time": min(durations),
            "max_time": max(durations),
            "operation_breakdown": operation_stats,
            "raw_data": self.performance_data
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Performance summary saved to: {summary_file}")
    
    def get_log_files(self) -> Dict[str, Path]:
        """Get all log file paths"""
        return {
            "text_log": self.text_log_file,
            "json_log": self.json_log_file,
            "error_log": self.error_log_file,
            "performance_log": self.perf_log_file
        }


def setup_global_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup global logging configuration"""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    # Create timestamped log file for global logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_log_file = log_dir_path / f"global_{timestamp}.log"
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(global_log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return global_log_file
