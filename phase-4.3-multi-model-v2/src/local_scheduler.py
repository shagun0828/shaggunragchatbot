"""
Local Scheduler Trigger for Phase 4.3 Multi-Model Ingest Pipeline
Comprehensive logging and monitoring for all pipeline phases
"""

import asyncio
import logging
import os
import sys
import json
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import subprocess

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_loader import EnvLoader
from utils.notifications import NotificationManager
from daily_ingest_pipeline import DailyIngestPipeline
from generate_pipeline_report import PipelineReportGenerator
from verify_chroma_upload import ChromaUploadVerifier
from health_check import HealthChecker
from send_notification import NotificationSender


@dataclass
class PipelinePhase:
    """Pipeline phase tracking"""
    name: str
    start_time: float
    end_time: float = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: str = None
    metrics: Dict[str, Any] = None
    duration: float = None


@dataclass
class SchedulerRun:
    """Complete scheduler run tracking"""
    run_id: str
    start_time: float
    end_time: float = None
    status: str = "running"
    phases: List[PipelinePhase] = None
    total_duration: float = None
    success_count: int = 0
    error_count: int = 0
    summary: Dict[str, Any] = None


class LocalSchedulerLogger:
    """Enhanced logging system for scheduler"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"scheduler_run_{timestamp}.log"
        self.json_log_file = self.log_dir / f"scheduler_run_{timestamp}.json"
        
        # Setup file logger
        self.logger = self._setup_logger()
        
        # JSON log entries
        self.json_entries = []
        
        self.logger.info(f"Local Scheduler Logger initialized - Log file: {self.log_file}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logger"""
        logger = logging.getLogger("LocalScheduler")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_phase_start(self, phase_name: str, metrics: Dict[str, Any] = None):
        """Log phase start"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "phase_start",
            "phase": phase_name,
            "metrics": metrics or {}
        }
        self.json_entries.append(entry)
        self.logger.info(f"Starting phase: {phase_name}")
        if metrics:
            self.logger.info(f"Phase metrics: {json.dumps(metrics, indent=2)}")
    
    def log_phase_end(self, phase_name: str, status: str, duration: float, metrics: Dict[str, Any] = None, error: str = None):
        """Log phase completion"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "phase_end",
            "phase": phase_name,
            "status": status,
            "duration": duration,
            "metrics": metrics or {},
            "error": error
        }
        self.json_entries.append(entry)
        
        if status == "completed":
            self.logger.info(f"Phase {phase_name} completed successfully in {duration:.2f}s")
        else:
            self.logger.error(f"Phase {phase_name} failed after {duration:.2f}s: {error}")
        
        if metrics:
            self.logger.info(f"Phase {phase_name} metrics: {json.dumps(metrics, indent=2)}")
    
    def log_system_info(self):
        """Log system information"""
        import psutil
        import platform
        
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage('/').total,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"System Information: {json.dumps(system_info, indent=2)}")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "system_info",
            "info": system_info
        }
        self.json_entries.append(entry)
    
    def log_environment(self, env_loader: EnvLoader):
        """Log environment configuration"""
        env_info = {
            "chroma_enabled": env_loader.get_bool("ENABLE_CHROMA_CLOUD", False),
            "chroma_tenant": env_loader.get_str("CHROMA_TENANT", "N/A"),
            "chroma_database": env_loader.get_str("CHROMA_DATABASE", "N/A"),
            "openai_enabled": bool(env_loader.get_str("OPENAI_API_KEY")),
            "embedding_model": env_loader.get_str("EMBEDDING_MODEL", "N/A"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Environment Configuration: {json.dumps(env_info, indent=2)}")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "environment",
            "config": env_info
        }
        self.json_entries.append(entry)
    
    def save_json_log(self):
        """Save JSON log entries to file"""
        with open(self.json_log_file, 'w') as f:
            json.dump(self.json_entries, f, indent=2)
        self.logger.info(f"JSON log saved to: {self.json_log_file}")
    
    def log_summary(self, scheduler_run: SchedulerRun):
        """Log final summary"""
        summary = {
            "run_id": scheduler_run.run_id,
            "total_duration": scheduler_run.total_duration,
            "status": scheduler_run.status,
            "success_count": scheduler_run.success_count,
            "error_count": scheduler_run.error_count,
            "phases": [asdict(phase) for phase in scheduler_run.phases],
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Scheduler Run Summary: {json.dumps(summary, indent=2)}")
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "run_summary",
            "summary": summary
        }
        self.json_entries.append(entry)


class LocalScheduler:
    """Local scheduler with comprehensive logging and monitoring"""
    
    def __init__(self):
        self.logger = LocalSchedulerLogger()
        self.env_loader = EnvLoader()
        self.notification_manager = NotificationManager()
        self.current_run = None
        
        # Initialize pipeline components
        self.pipeline = DailyIngestPipeline()
        self.report_generator = PipelineReportGenerator()
        self.upload_verifier = ChromaUploadVerifier()
        self.health_checker = HealthChecker()
        self.notification_sender = NotificationSender()
        
        self.logger.log_system_info()
        self.logger.log_environment(self.env_loader)
    
    async def run_complete_pipeline(self) -> SchedulerRun:
        """Run complete ingest pipeline with comprehensive logging"""
        run_id = f"local_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = SchedulerRun(
            run_id=run_id,
            start_time=time.time(),
            phases=[]
        )
        
        self.logger.logger.info(f"Starting complete pipeline run: {run_id}")
        
        try:
            # Phase 1: Health Check
            await self._run_phase("health_check", self._health_check_phase)
            
            # Phase 2: Data Ingestion
            await self._run_phase("data_ingestion", self._data_ingestion_phase)
            
            # Phase 3: Chroma Upload Verification
            await self._run_phase("upload_verification", self._upload_verification_phase)
            
            # Phase 4: Report Generation
            await self._run_phase("report_generation", self._report_generation_phase)
            
            # Phase 5: Notification
            await self._run_phase("notification", self._notification_phase)
            
            # Final summary
            self.current_run.end_time = time.time()
            self.current_run.total_duration = self.current_run.end_time - self.current_run.start_time
            self.current_run.status = "completed" if self.current_run.error_count == 0 else "partial_success"
            
            self.logger.log_summary(self.current_run)
            
        except Exception as e:
            self.current_run.end_time = time.time()
            self.current_run.total_duration = self.current_run.end_time - self.current_run.start_time
            self.current_run.status = "failed"
            self.current_run.error_count += 1
            
            self.logger.logger.error(f"Pipeline run failed: {str(e)}")
            self.logger.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try to send error notification
            try:
                await self._send_error_notification(str(e))
            except Exception as notification_error:
                self.logger.logger.error(f"Failed to send error notification: {notification_error}")
        
        finally:
            # Save logs
            self.logger.save_json_log()
            
        return self.current_run
    
    async def _run_phase(self, phase_name: str, phase_func):
        """Run a single phase with tracking"""
        phase = PipelinePhase(
            name=phase_name,
            start_time=time.time(),
            status="running"
        )
        
        self.current_run.phases.append(phase)
        self.logger.log_phase_start(phase_name)
        
        try:
            metrics = await phase_func()
            phase.end_time = time.time()
            phase.duration = phase.end_time - phase.start_time
            phase.status = "completed"
            phase.metrics = metrics
            
            self.current_run.success_count += 1
            
            self.logger.log_phase_end(
                phase_name, 
                "completed", 
                phase.duration, 
                metrics
            )
            
        except Exception as e:
            phase.end_time = time.time()
            phase.duration = phase.end_time - phase.start_time
            phase.status = "failed"
            phase.error_message = str(e)
            
            self.current_run.error_count += 1
            
            self.logger.log_phase_end(
                phase_name, 
                "failed", 
                phase.duration, 
                error=str(e)
            )
            
            # Decide whether to continue or stop
            if phase_name in ["health_check", "data_ingestion"]:
                raise  # Critical phase failure, stop pipeline
            else:
                self.logger.logger.warning(f"Phase {phase_name} failed, continuing pipeline")
    
    async def _health_check_phase(self) -> Dict[str, Any]:
        """Phase 1: Health Check"""
        self.logger.logger.info("Running health check phase...")
        
        health_status = await self.health_checker.check_all_components()
        
        metrics = {
            "overall_health": health_status.get("overall_health", "unknown"),
            "components_checked": len(health_status.get("components", {})),
            "healthy_components": len([
                comp for comp in health_status.get("components", {}).values()
                if comp.get("status") == "healthy"
            ]),
            "unhealthy_components": len([
                comp for comp in health_status.get("components", {}).values()
                if comp.get("status") != "healthy"
            ])
        }
        
        # Check if critical components are healthy
        critical_components = ["chroma_client", "llm_client", "database"]
        for component in critical_components:
            if component in health_status.get("components", {}):
                comp_status = health_status["components"][component].get("status")
                if comp_status != "healthy":
                    raise Exception(f"Critical component {component} is {comp_status}")
        
        self.logger.logger.info("Health check completed successfully")
        return metrics
    
    async def _data_ingestion_phase(self) -> Dict[str, Any]:
        """Phase 2: Data Ingestion"""
        self.logger.logger.info("Running data ingestion phase...")
        
        # Run the daily ingest pipeline
        ingestion_result = await self.pipeline.run_complete_pipeline()
        
        metrics = {
            "scraped_articles": ingestion_result.get("scraped_articles", 0),
            "processed_articles": ingestion_result.get("processed_articles", 0),
            "embedded_documents": ingestion_result.get("embedded_documents", 0),
            "uploaded_documents": ingestion_result.get("uploaded_documents", 0),
            "failed_uploads": ingestion_result.get("failed_uploads", 0),
            "processing_time": ingestion_result.get("total_processing_time", 0),
            "errors": len(ingestion_result.get("errors", []))
        }
        
        # Check if ingestion was successful
        if metrics["uploaded_documents"] == 0 and metrics["scraped_articles"] > 0:
            raise Exception("No documents were uploaded despite scraping articles")
        
        self.logger.logger.info("Data ingestion completed successfully")
        return metrics
    
    async def _upload_verification_phase(self) -> Dict[str, Any]:
        """Phase 3: Upload Verification"""
        self.logger.logger.info("Running upload verification phase...")
        
        verification_result = await self.upload_verifier.verify_upload()
        
        metrics = {
            "total_verified": verification_result.get("total_verified", 0),
            "verification_passed": verification_result.get("verification_passed", 0),
            "verification_failed": verification_result.get("verification_failed", 0),
            "chroma_collection_size": verification_result.get("chroma_collection_size", 0),
            "sample_queries_tested": verification_result.get("sample_queries_tested", 0),
            "avg_similarity_score": verification_result.get("avg_similarity_score", 0)
        }
        
        # Check if verification passed
        if metrics["verification_failed"] > metrics["verification_passed"]:
            raise Exception("Upload verification failed for majority of documents")
        
        self.logger.logger.info("Upload verification completed successfully")
        return metrics
    
    async def _report_generation_phase(self) -> Dict[str, Any]:
        """Phase 4: Report Generation"""
        self.logger.logger.info("Running report generation phase...")
        
        # Generate pipeline report
        report_result = await self.report_generator.generate_report()
        
        metrics = {
            "report_generated": report_result.get("report_generated", False),
            "report_file": report_result.get("report_file", ""),
            "charts_generated": report_result.get("charts_generated", 0),
            "summary_stats": report_result.get("summary_stats", {}),
            "error_count": report_result.get("error_count", 0)
        }
        
        if not metrics["report_generated"]:
            raise Exception("Failed to generate pipeline report")
        
        self.logger.logger.info("Report generation completed successfully")
        return metrics
    
    async def _notification_phase(self) -> Dict[str, Any]:
        """Phase 5: Notification"""
        self.logger.logger.info("Running notification phase...")
        
        # Send completion notification
        notification_result = await self.notification_sender.send_completion_notification()
        
        metrics = {
            "notification_sent": notification_result.get("sent", False),
            "notification_method": notification_result.get("method", ""),
            "recipients": notification_result.get("recipients", []),
            "delivery_status": notification_result.get("delivery_status", "")
        }
        
        self.logger.logger.info("Notification phase completed")
        return metrics
    
    async def _send_error_notification(self, error_message: str):
        """Send error notification"""
        try:
            await self.notification_sender.send_error_notification(error_message)
        except Exception as e:
            self.logger.logger.error(f"Failed to send error notification: {e}")
    
    def print_summary(self):
        """Print detailed summary of the run"""
        if not self.current_run:
            print("No run data available")
            return
        
        print("\n" + "="*80)
        print("LOCAL SCHEDULER RUN SUMMARY")
        print("="*80)
        
        print(f"Run ID: {self.current_run.run_id}")
        print(f"Status: {self.current_run.status}")
        print(f"Total Duration: {self.current_run.total_duration:.2f} seconds")
        print(f"Success Count: {self.current_run.success_count}")
        print(f"Error Count: {self.current_run.error_count}")
        
        print("\nPhase Details:")
        print("-" * 40)
        
        for phase in self.current_run.phases:
            status_icon = "SUCCESS" if phase.status == "completed" else "FAILED"
            print(f"{phase.name:20} | {status_icon:8} | {phase.duration:8.2f}s")
            if phase.error_message:
                print(f"{'':20} | Error: {phase.error_message}")
        
        print("\nLog Files:")
        print("-" * 40)
        print(f"Text Log: {self.logger.log_file}")
        print(f"JSON Log: {self.logger.json_log_file}")
        
        print("\n" + "="*80)


async def main():
    """Main function to run local scheduler"""
    print("Starting Local Scheduler for Phase 4.3 Multi-Model Ingest Pipeline")
    print("="*80)
    
    scheduler = LocalScheduler()
    
    try:
        # Run complete pipeline
        result = await scheduler.run_complete_pipeline()
        
        # Print summary
        scheduler.print_summary()
        
        # Return appropriate exit code
        if result.status == "completed":
            print("\nPipeline completed successfully!")
            sys.exit(0)
        elif result.status == "partial_success":
            print("\nPipeline completed with some errors!")
            sys.exit(1)
        else:
            print("\nPipeline failed!")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nPipeline failed with exception: {e}")
        print(f"Check log files for detailed error information")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
