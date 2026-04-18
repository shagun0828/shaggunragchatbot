#!/usr/bin/env python3
"""
Main entry point for the mutual fund scraping service.
Phase 4.0: Scheduler and Scraping Service
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from scrapers.mutual_fund_scraper import MutualFundScraper
from processors.data_processor import DataProcessor
from storage.vector_storage import VectorStorage
from utils.logger import setup_logger
from utils.notifications import NotificationManager


async def main():
    """Main scraping service execution"""
    # Setup logging
    logger = setup_logger("mutual_fund_scraper")
    logger.info("Starting mutual fund scraping service")
    
    try:
        # Initialize components
        scraper = MutualFundScraper()
        processor = DataProcessor()
        vector_storage = VectorStorage()
        notification_manager = NotificationManager()
        
        # Define Groww URLs to scrape
        urls = [
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
        ]
        
        logger.info(f"Scraping {len(urls)} mutual fund URLs")
        
        # Step 1: Scrape data from URLs
        raw_data = await scraper.scrape_all_funds(urls)
        logger.info(f"Scraped data from {len(raw_data)} funds")
        
        # Step 2: Process and validate data
        processed_data = await processor.process_scraped_data(raw_data)
        logger.info(f"Processed {len(processed_data)} fund records")
        
        # Step 3: Generate embeddings and store in vector database
        if os.getenv('VECTOR_DB_URL') and os.getenv('VECTOR_DB_API_KEY'):
            await vector_storage.store_embeddings(processed_data)
            logger.info("Successfully stored embeddings in vector database")
        else:
            logger.warning("Vector database credentials not provided, skipping vector storage")
        
        # Step 4: Save processed data locally
        await processor.save_processed_data(processed_data)
        logger.info("Saved processed data locally")
        
        # Step 5: Send success notification
        await notification_manager.send_success_notification(
            f"Successfully scraped and processed {len(processed_data)} mutual funds"
        )
        
        logger.info("Mutual fund scraping service completed successfully")
        
    except Exception as e:
        logger.error(f"Scraping service failed: {str(e)}")
        
        # Send failure notification
        try:
            notification_manager = NotificationManager()
            await notification_manager.send_failure_notification(
                f"Scraping service failed: {str(e)}"
            )
        except Exception as notification_error:
            logger.error(f"Failed to send notification: {str(notification_error)}")
        
        # Re-raise to fail the GitHub Actions job
        raise


if __name__ == "__main__":
    asyncio.run(main())
