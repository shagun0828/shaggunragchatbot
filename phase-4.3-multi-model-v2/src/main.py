#!/usr/bin/env python3
"""
Main entry point for Phase 4.3: Multi-Model Processing v2
Comprehensive implementation with BGE-base and BGE-small coordination
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from processors.multi_model_coordinator import MultiModelCoordinator, CoordinationMode, ProcessingStrategy
from routers.intelligent_url_router import ModelType, ContentType
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from utils.notifications import NotificationManager
from utils.data_simulator import DataSimulator


class MultiModelSystem:
    """Main system for Phase 4.3 multi-model processing"""
    
    def __init__(self, config_path: str = "config/multi_model_config.yaml"):
        self.logger = setup_logger("multi_model_system")
        self.config = ConfigLoader.load_config(config_path)
        
        # Initialize multi-model coordinator
        self.coordinator = MultiModelCoordinator(self.config.get('multi_model', {}))
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
            'model_utilization': {},
            'coordination_efficiency': 0.0
        }
    
    async def start(self) -> None:
        """Start the multi-model system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting Phase 4.3 Multi-Model Processing System v2")
        self.is_running = True
        self.start_time = time.time()
        
        # Send startup notification
        await self.notification_manager.send_success_notification(
            "Phase 4.3 Multi-Model System v2 started successfully"
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
        """Process a list of URLs using intelligent multi-model coordination"""
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
                    'source': 'multi_model_system_v2'
                }
            })
        
        # Process with multi-model coordinator
        result = await self.coordinator.process_urls(url_data)
        
        # Update system metrics
        self.system_metrics['total_sessions'] += 1
        self.system_metrics['total_urls_processed'] += result['total_urls']
        self.system_metrics['avg_throughput'] = result['total_chunks'] / result['coordination_metadata']['processing_time']
        self.system_metrics['model_utilization'] = result['coordination_metadata']['model_utilization']
        self.system_metrics['coordination_efficiency'] = result['coordination_metadata']['routing_efficiency']
        
        return result
    
    async def process_sample_data(self) -> Dict[str, Any]:
        """Process sample data for comprehensive demonstration"""
        # Sample URLs for demonstration (20 mutual funds + 5 financial news)
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
            'component_metrics': self.coordinator.get_comprehensive_metrics(),
            'coordination_mode': self.coordinator.coordination_mode.value,
            'processing_strategy': self.coordinator.processing_strategy.value,
            'model_limits': {
                'bge_base_max_urls': 20,
                'bge_small_max_urls': 5,
                'total_capacity': 25
            }
        }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return await self.coordinator.generate_performance_report()
    
    def set_coordination_mode(self, mode: CoordinationMode) -> None:
        """Set coordination mode"""
        self.coordinator.set_coordination_mode(mode)
        self.logger.info(f"Coordination mode set to: {mode.value}")
    
    def set_processing_strategy(self, strategy: ProcessingStrategy) -> None:
        """Set processing strategy"""
        self.coordinator.set_processing_strategy(strategy)
        self.logger.info(f"Processing strategy set to: {strategy.value}")
    
    async def demonstrate_coordination_modes(self) -> Dict[str, Any]:
        """Demonstrate different coordination modes"""
        self.logger.info("Demonstrating different coordination modes")
        
        # Sample URLs for demonstration
        sample_urls = [
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
            "https://www.economictimes.com/markets/stocks/news",
            "https://www.livemint.com/market/stock-market-news",
            "https://moneycontrol.com/news"
        ]
        
        results = {}
        
        # Test different coordination modes
        for mode in [CoordinationMode.BALANCED, CoordinationMode.QUALITY_FOCUSED, CoordinationMode.SPEED_FOCUSED]:
            self.logger.info(f"Testing {mode.value} coordination mode")
            
            self.set_coordination_mode(mode)
            result = await self.process_urls(sample_urls)
            
            results[mode.value] = {
                'avg_quality': result.get('quality_comparison', {}).get('bge_base_quality', 0),
                'processing_time': result['coordination_metadata']['processing_time'],
                'routing_efficiency': result['coordination_metadata']['routing_efficiency'],
                'model_utilization': result['coordination_metadata']['model_utilization']
            }
        
        return results
    
    async def demonstrate_processing_strategies(self) -> Dict[str, Any]:
        """Demonstrate different processing strategies"""
        self.logger.info("Demonstrating different processing strategies")
        
        # Sample URLs for demonstration
        sample_urls = [
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth",
            "https://www.economictimes.com/markets/stocks/news",
            "https://www.livemint.com/market/stock-market-news"
        ]
        
        results = {}
        
        # Test different processing strategies
        for strategy in [ProcessingStrategy.SEQUENTIAL, ProcessingStrategy.PARALLEL, ProcessingStrategy.HYBRID]:
            self.logger.info(f"Testing {strategy.value} processing strategy")
            
            self.set_processing_strategy(strategy)
            result = await self.process_urls(sample_urls)
            
            results[strategy.value] = {
                'processing_time': result['coordination_metadata']['processing_time'],
                'coordination_overhead': result['coordination_metadata']['coordination_overhead'],
                'throughput': result['total_chunks'] / result['coordination_metadata']['processing_time'],
                'strategy_used': result['coordination_metadata']['processing_strategy']
            }
        
        return results


async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Phase 4.3 Multi-Model Processing System v2")
    
    # Initialize system
    system = MultiModelSystem()
    
    try:
        # Start system
        await system.start()
        
        # Process sample data
        logger.info("Processing sample data with intelligent multi-model coordination...")
        result = await system.process_sample_data()
        
        # Display results
        logger.info("Multi-Model Processing Results:")
        logger.info(f"  Total URLs: {result['total_urls']}")
        logger.info(f"  BGE-base URLs: {result['bge_base_results']['urls_processed']}")
        logger.info(f"  BGE-small URLs: {result['bge_small_results']['urls_processed']}")
        logger.info(f"  Total Chunks: {result['total_chunks']}")
        logger.info(f"  Total Embeddings: {result['total_embeddings']}")
        logger.info(f"  Processing Time: {result['coordination_metadata']['processing_time']:.2f}s")
        logger.info(f"  Routing Summary: {result['routing_summary']}")
        
        # Quality comparison
        if 'quality_comparison' in result:
            quality_comp = result['quality_comparison']
            logger.info(f"  Quality Comparison:")
            logger.info(f"    BGE-base Quality: {quality_comp.get('bge_base_quality', 0):.3f}")
            logger.info(f"    BGE-small Quality: {quality_comp.get('bge_small_quality', 0):.3f}")
            logger.info(f"    Better Model: {quality_comp.get('better_model', 'unknown')}")
            logger.info(f"    Quality Consistency: {quality_comp.get('quality_consistency', False)}")
        
        # Coordination metadata
        coord_meta = result['coordination_metadata']
        logger.info(f"\nCoordination Metadata:")
        logger.info(f"  Processing Strategy: {coord_meta['processing_strategy']}")
        logger.info(f"  Coordination Mode: {coord_meta['coordination_mode']}")
        logger.info(f"  Routing Efficiency: {coord_meta['routing_efficiency']:.3f}")
        logger.info(f"  Coordination Overhead: {coord_meta['coordination_overhead']:.3f}")
        logger.info(f"  Model Utilization: {coord_meta['model_utilization']}")
        logger.info(f"  Quality Consistency: {coord_meta['quality_consistency']:.3f}")
        
        # Performance metrics
        if 'performance_metrics' in result:
            perf = result['performance_metrics']
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Total Processing Time: {perf.get('total_processing_time', 0):.2f}s")
            logger.info(f"  Average Throughput: {perf.get('avg_throughput', 0):.2f} chunks/s")
            
            if 'model_efficiency' in perf:
                for model, metrics in perf['model_efficiency'].items():
                    logger.info(f"  {model.title()} Efficiency:")
                    logger.info(f"    Throughput: {metrics.get('throughput', 0):.2f} chunks/s")
                    logger.info(f"    Quality Score: {metrics.get('quality_score', 0):.3f}")
                    logger.info(f"    Dimension: {metrics.get('dimension', 0)}")
        
        # Get system status
        status = await system.get_system_status()
        logger.info(f"\nSystem Status:")
        logger.info(f"  Uptime: {status['system_status']['uptime']:.2f}s")
        logger.info(f"  Total Sessions: {status['system_metrics']['total_sessions']}")
        logger.info(f"  Total URLs Processed: {status['system_metrics']['total_urls_processed']}")
        logger.info(f"  Average Throughput: {status['system_metrics']['avg_throughput']:.2f} chunks/s")
        logger.info(f"  Coordination Efficiency: {status['system_metrics']['coordination_efficiency']:.3f}")
        
        # Demonstrate coordination modes
        logger.info("\nDemonstrating different coordination modes...")
        coordination_results = await system.demonstrate_coordination_modes()
        
        logger.info("Coordination Mode Comparison:")
        for mode, metrics in coordination_results.items():
            logger.info(f"  {mode.title()}:")
            logger.info(f"    Avg Quality: {metrics['avg_quality']:.3f}")
            logger.info(f"    Processing Time: {metrics['processing_time']:.2f}s")
            logger.info(f"    Routing Efficiency: {metrics['routing_efficiency']:.3f}")
        
        # Demonstrate processing strategies
        logger.info("\nDemonstrating different processing strategies...")
        strategy_results = await system.demonstrate_processing_strategies()
        
        logger.info("Processing Strategy Comparison:")
        for strategy, metrics in strategy_results.items():
            logger.info(f"  {strategy.title()}:")
            logger.info(f"    Processing Time: {metrics['processing_time']:.2f}s")
            logger.info(f"    Throughput: {metrics['throughput']:.2f} chunks/s")
            logger.info(f"    Coordination Overhead: {metrics['coordination_overhead']:.3f}")
        
        # Generate performance report
        logger.info("\nGenerating comprehensive performance report...")
        performance_report = await system.generate_performance_report()
        
        logger.info("Performance Report Summary:")
        logger.info(f"  System Health: {performance_report['system_health']['health_status']}")
        logger.info(f"  Health Score: {performance_report['system_health']['health_score']:.3f}")
        
        if 'recommendations' in performance_report:
            logger.info("  Recommendations:")
            for i, rec in enumerate(performance_report['recommendations'], 1):
                logger.info(f"    {i}. {rec}")
        
        # Advantages demonstrated
        logger.info("\nMulti-Model Advantages Demonstrated:")
        logger.info("  â Intelligent Routing: Automatic model selection based on URL analysis")
        logger.info("  â Adaptive Processing: Dynamic strategy selection for optimal performance")
        logger.info("  â Quality Management: Comprehensive quality assessment and consistency")
        logger.info("  â Resource Optimization: Efficient utilization of BGE-base and BGE-small")
        logger.info("  â Coordination Efficiency: Minimal overhead with optimal routing")
        logger.info("  â Performance Monitoring: Real-time metrics and health assessment")
        logger.info("  â Flexible Configuration: Multiple coordination modes and strategies")
        
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
        logger.info("Phase 4.3 Multi-Model Processing System v2 completed")


if __name__ == "__main__":
    asyncio.run(main())
