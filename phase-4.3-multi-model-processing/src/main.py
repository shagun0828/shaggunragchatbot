#!/usr/bin/env python3
"""
Main entry point for Phase 4.3: Multi-Model Processing
Coordinates BGE-base and BGE-small models for optimal URL processing
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from processors.multi_model_processor import MultiModelProcessor
from routers.url_model_router import URLModelRouter
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from utils.notifications import NotificationManager
from utils.data_simulator import DataSimulator


class MultiModelSystem:
    """Main system for Phase 4.3 multi-model processing"""
    
    def __init__(self, config_path: str = "config/multi_model_config.yaml"):
        self.logger = setup_logger("multi_model_system")
        self.config = ConfigLoader.load_config(config_path)
        
        # Initialize multi-model processor
        self.multi_model_processor = MultiModelProcessor(self.config.get('multi_model', {}))
        self.notification_manager = NotificationManager()
        self.data_simulator = DataSimulator()
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Performance metrics
        self.system_metrics = {
            'total_sessions': 0,
            'total_urls_processed': 0,
            'total_processing_time': 0.0,
            'avg_throughput': 0.0,
            'model_utilization': {}
        }
    
    async def start(self) -> None:
        """Start the multi-model system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting Phase 4.3 Multi-Model Processing System")
        self.is_running = True
        self.start_time = time.time()
        
        # Send startup notification
        await self.notification_manager.send_success_notification(
            "Phase 4.3 Multi-Model System started successfully"
        )
        
        self.logger.info("Multi-model system started")
    
    async def stop(self) -> None:
        """Stop the multi-model system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping multi-model system")
        self.is_running = False
        
        # Calculate final metrics
        uptime = time.time() - self.start_time if self.start_time else 0
        self.system_metrics['total_processing_time'] = uptime
        
        # Send shutdown notification
        await self.notification_manager.send_success_notification(
            f"Multi-model system stopped. Uptime: {uptime:.2f}s, "
            f"Total URLs processed: {self.system_metrics['total_urls_processed']}"
        )
        
        self.logger.info(f"Multi-model system stopped after {uptime:.2f} seconds")
    
    async def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Process a list of URLs using optimal model selection"""
        self.logger.info(f"Processing {len(urls)} URLs with multi-model system")
        
        # Simulate URL data (in production, this would come from scraping)
        url_data = []
        for i, url in enumerate(urls):
            content = self.data_simulator.generate_content_for_url(url)
            url_data.append({
                'url': url,
                'content': content,
                'metadata': {
                    'index': i,
                    'timestamp': time.time(),
                    'source': 'multi_model_system'
                }
            })
        
        # Process with multi-model processor
        result = await self.multi_model_processor.process_urls(url_data)
        
        # Update system metrics
        self.system_metrics['total_sessions'] += 1
        self.system_metrics['total_urls_processed'] += result['total_urls']
        self.system_metrics['avg_throughput'] = result['total_chunks'] / result['processing_metadata']['total_processing_time']
        self.system_metrics['model_utilization'] = result['processing_metadata']['model_utilization']
        
        return result
    
    async def process_sample_data(self) -> Dict[str, Any]:
        """Process sample data for demonstration"""
        # Sample URLs for demonstration
        sample_urls = [
            # Mutual fund URLs (should use BGE-base)
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-technology-fund-direct-growth",
            "https://groww.in/mutual-funds/axis-bluechip-fund-direct-growth",
            "https://groww.in/mutual-funds/sbi-small-cap-fund-direct-plan-growth",
            "https://groww.in/mutual-funds/uti-nifty-index-fund-direct-growth",
            "https://groww.in/mutual-funds/mirae-asset-large-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/tata-digital-india-fund-direct-growth",
            "https://groww.in/mutual-funds/kotak-emerging-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/franklin-india-prima-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-balanced-advantage-fund-direct-growth",
            "https://groww.in/mutual-funds/axis-hybrid-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/sbi-balanced-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-arbitrage-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-equity-arbitrage-fund-direct-growth",
            "https://groww.in/mutual-funds/nippon-india-growth-fund-direct-growth",
            
            # Financial news URLs (should use BGE-small)
            "https://www.economictimes.com/markets/stocks/news",
            "https://www.livemint.com/market/stock-market-news",
            "https://www.business-standard.com/markets",
            "https://www.financial-express.com/market",
            "https://moneycontrol.com/news"
        ]
        
        return await self.process_urls(sample_urls)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_status': {
                'is_running': self.is_running,
                'uptime': time.time() - self.start_time if self.start_time else 0,
                'start_time': self.start_time
            },
            'system_metrics': self.system_metrics,
            'component_metrics': self.multi_model_processor.get_comprehensive_metrics(),
            'model_limits': {
                'bge_base_max_urls': 20,
                'bge_small_max_urls': 5,
                'total_capacity': 25
            }
        }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        comparison_report = await self.multi_model_processor.generate_comparison_report()
        
        report = {
            'report_timestamp': time.time(),
            'system_metrics': self.system_metrics,
            'model_comparison': comparison_report,
            'routing_efficiency': self.multi_model_processor.url_router.get_routing_statistics(),
            'recommendations': comparison_report.get('recommendations', []),
            'performance_summary': {
                'total_urls_processed': self.system_metrics['total_urls_processed'],
                'avg_throughput': self.system_metrics['avg_throughput'],
                'model_distribution': self.system_metrics['model_utilization'],
                'efficiency_score': self._calculate_efficiency_score()
            }
        }
        
        return report
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        if not self.system_metrics['model_utilization']:
            return 0.0
        
        # Calculate utilization efficiency
        base_util = self.system_metrics['model_utilization'].get('bge_base', 0)
        small_util = self.system_metrics['model_utilization'].get('bge_small', 0)
        
        # Weighted utilization (base model is more important)
        utilization_score = (base_util * 0.7 + small_util * 0.3)
        
        # Throughput score
        throughput_score = min(self.system_metrics['avg_throughput'] / 100, 1.0)  # Normalize to 0-1
        
        # Combined efficiency score
        efficiency_score = (utilization_score * 0.6 + throughput_score * 0.4)
        
        return efficiency_score


async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Phase 4.3 Multi-Model Processing System")
    
    # Initialize system
    system = MultiModelSystem()
    
    try:
        # Start system
        await system.start()
        
        # Process sample data
        logger.info("Processing sample data with multi-model system...")
        result = await system.process_sample_data()
        
        # Display results
        logger.info("Multi-Model Processing Results:")
        logger.info(f"  Total URLs: {result['total_urls']}")
        logger.info(f"  BGE-base URLs: {result['bge_base_results']['urls_processed']}")
        logger.info(f"  BGE-small URLs: {result['bge_small_results']['urls_processed']}")
        logger.info(f"  Total Chunks: {result['total_chunks']}")
        logger.info(f"  Total Embeddings: {result['total_embeddings']}")
        logger.info(f"  Processing Time: {result['processing_metadata']['total_processing_time']:.2f}s")
        logger.info(f"  Routing Summary: {result['routing_summary']}")
        
        # Quality comparison
        if 'quality_comparison' in result:
            quality_comp = result['quality_comparison']
            logger.info(f"  Quality Comparison:")
            logger.info(f"    BGE-base Quality: {quality_comp.get('bge_base_quality', 0):.3f}")
            logger.info(f"    BGE-small Quality: {quality_comp.get('bge_small_quality', 0):.3f}")
            logger.info(f"    Better Model: {quality_comp.get('better_model', 'unknown')}")
        
        # Efficiency metrics
        if 'efficiency_metrics' in result:
            efficiency = result['efficiency_metrics']
            logger.info(f"  Efficiency Metrics:")
            logger.info(f"    Total Processing Time: {efficiency.get('total_processing_time', 0):.2f}s")
            logger.info(f"    Average Throughput: {efficiency.get('avg_throughput', 0):.2f} chunks/s")
            logger.info(f"    Model Utilization: {result['processing_metadata']['model_utilization']}")
        
        # Get system status
        status = await system.get_system_status()
        logger.info(f"\nSystem Status:")
        logger.info(f"  Uptime: {status['system_status']['uptime']:.2f}s")
        logger.info(f"  Total Sessions: {status['system_metrics']['total_sessions']}")
        logger.info(f"  Total URLs Processed: {status['system_metrics']['total_urls_processed']}")
        logger.info(f"  Average Throughput: {status['system_metrics']['avg_throughput']:.2f} chunks/s")
        
        # Generate performance report
        logger.info("\nGenerating performance report...")
        performance_report = await system.generate_performance_report()
        
        logger.info("Performance Report Summary:")
        logger.info(f"  Efficiency Score: {performance_report['performance_summary']['efficiency_score']:.3f}")
        logger.info(f"  Model Distribution: {performance_report['performance_summary']['model_distribution']}")
        
        if performance_report['recommendations']:
            logger.info("  Recommendations:")
            for i, rec in enumerate(performance_report['recommendations'], 1):
                logger.info(f"    {i}. {rec}")
        
        # Demonstrate model advantages
        logger.info("\nModel Advantages Demonstrated:")
        logger.info("  â BGE-base: Higher quality (768 dimensions) for complex financial data")
        logger.info("  â BGE-small: Faster processing (384 dimensions) for simpler content")
        logger.info("  â Smart Routing: Automatic model selection based on URL analysis")
        logger.info("  â Efficient Resource Use: Optimal model utilization within limits")
        logger.info("  â Cost Effective: Both models are local, no API costs")
        
        # Wait to show system stability
        logger.info("\nSystem running... (waiting 5 seconds)")
        await asyncio.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
    finally:
        # Stop system
        await system.stop()
        logger.info("Phase 4.3 Multi-Model Processing System completed")


if __name__ == "__main__":
    asyncio.run(main())
