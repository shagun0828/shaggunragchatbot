#!/usr/bin/env python3
"""
Quick Start Script for Simple Local Scheduler
Run this script to trigger the complete ingest pipeline locally
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_local_scheduler import SimpleLocalScheduler


def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("PHASE 4.3 MULTI-MODEL INGEST PIPELINE - SIMPLE LOCAL SCHEDULER")
    print("=" * 80)
    print("This script will run the complete ingest pipeline locally with:")
    print("1. Health Check - Verify all components are healthy")
    print("2. Data Ingestion - Scrape, process, and embed documents")
    print("3. Upload Verification - Verify Chroma Cloud upload")
    print("4. Report Generation - Generate comprehensive reports")
    print("5. Notification - Send completion notifications")
    print("=" * 80)
    print()


def check_prerequisites():
    """Check if prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please copy .env.example to .env and configure your settings")
        return False
    
    # Check if logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Check if data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if reports directory exists
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    print("Prerequisites check passed!")
    print()
    return True


async def main():
    """Main function"""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create and run scheduler
    print("Initializing simple local scheduler...")
    scheduler = SimpleLocalScheduler()
    
    print("Starting complete pipeline run...")
    print("This may take several minutes to complete.")
    print("Check the log files for detailed progress.")
    print()
    
    try:
        # Run the complete pipeline
        result = await scheduler.run_complete_pipeline()
        
        # Print summary
        scheduler.print_summary()
        
        # Print final status
        print("\n" + "="*80)
        if result.status == "completed":
            print("SUCCESS: All phases completed successfully!")
            print("Check the generated reports and log files for details.")
        elif result.status == "partial_success":
            print("PARTIAL SUCCESS: Some phases completed with errors.")
            print("Check the error log for details.")
        else:
            print("FAILURE: Pipeline failed.")
            print("Check the error log for details.")
        print("="*80)
        
        # Return appropriate exit code
        if result.status == "completed":
            return 0
        elif result.status == "partial_success":
            return 1
        else:
            return 2
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\nPipeline failed with exception: {e}")
        print("Check the log files for detailed error information")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
