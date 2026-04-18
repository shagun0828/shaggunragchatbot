#!/usr/bin/env python3
"""
Health Check System for Pipeline Components
Monitors and reports health of various pipeline components
"""

import argparse
import asyncio
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from simple_chroma_cloud import ChromaCloudManager
from utils.env_loader import EnvLoader


class HealthChecker:
    """Comprehensive health checker for pipeline components"""
    
    def __init__(self):
        self.env_loader = EnvLoader()
        self.components = {}
        self.start_time = time.time()
    
    async def check_component_health(self, component: str) -> dict:
        """Check health of a specific component"""
        print(f"Checking health of: {component}")
        
        if component == 'chroma_cloud':
            return await self._check_chroma_cloud_health()
        elif component == 'pipeline':
            return await self._check_pipeline_health()
        elif component == 'data_quality':
            return await self._check_data_quality_health()
        elif component == 'environment':
            return self._check_environment_health()
        else:
            return {
                'status': 'unknown',
                'message': f'Unknown component: {component}'
            }
    
    async def _check_chroma_cloud_health(self) -> dict:
        """Check Chroma Cloud health"""
        try:
            manager = ChromaCloudManager()
            
            # Check connection
            health = await manager.health_check()
            
            if health['status'] == 'healthy':
                # Get additional metrics
                metrics = manager.get_metrics()
                
                return {
                    'status': 'healthy',
                    'connection': health['connection'],
                    'collections_count': health['collections_count'],
                    'total_documents': metrics['current_metrics']['documents_uploaded'],
                    'last_upload': metrics['current_metrics']['last_upload_time'],
                    'upload_speed': metrics['current_metrics']['avg_upload_speed'],
                    'timestamp': time.time()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'connection': health['connection'],
                    'error': health.get('error', 'Unknown error'),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def _check_pipeline_health(self) -> dict:
        """Check pipeline health"""
        try:
            # Check recent pipeline runs
            today = datetime.now().strftime('%Y-%m-%d')
            pipeline_file = f"reports/daily_ingest_{today}.json"
            
            if Path(pipeline_file).exists():
                with open(pipeline_file, 'r') as f:
                    pipeline_data = json.load(f)
                
                status = pipeline_data.get('status', 'unknown')
                metrics = pipeline_data.get('metrics', {})
                
                # Calculate health score
                health_score = 100
                
                if status != 'completed':
                    health_score -= 50
                
                if metrics.get('error_count', 0) > 0:
                    health_score -= metrics['error_count'] * 10
                
                if metrics.get('scraping_errors', 0) > 5:
                    health_score -= 20
                
                health_status = 'healthy' if health_score >= 80 else 'warning' if health_score >= 60 else 'unhealthy'
                
                return {
                    'status': health_status,
                    'health_score': health_score,
                    'last_run': pipeline_data.get('timestamp'),
                    'last_status': status,
                    'errors': metrics.get('error_count', 0),
                    'scraping_errors': metrics.get('scraping_errors', 0),
                    'upload_count': metrics.get('documents_uploaded', 0),
                    'timestamp': time.time()
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'No pipeline run found for today',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def _check_data_quality_health(self) -> dict:
        """Check data quality health"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            pipeline_file = f"reports/daily_ingest_{today}.json"
            
            if Path(pipeline_file).exists():
                with open(pipeline_file, 'r') as f:
                    pipeline_data = json.load(f)
                
                scraping = pipeline_data.get('scraping_results', {})
                chunking = pipeline_data.get('chunking_results', {})
                
                # Calculate quality metrics
                urls_scraped = scraping.get('urls_scraped', 0)
                scraping_errors = scraping.get('scraping_errors', 0)
                avg_chunk_size = chunking.get('avg_chunk_size', 0)
                
                # Quality score
                quality_score = 100
                
                if urls_scraped < 20:
                    quality_score -= 30
                
                if scraping_errors > 0:
                    quality_score -= scraping_errors * 5
                
                if avg_chunk_size < 100 or avg_chunk_size > 1500:
                    quality_score -= 20
                
                quality_status = 'excellent' if quality_score >= 90 else 'good' if quality_score >= 70 else 'fair' if quality_score >= 50 else 'poor'
                
                return {
                    'status': quality_status,
                    'quality_score': quality_score,
                    'urls_scraped': urls_scraped,
                    'scraping_errors': scraping_errors,
                    'avg_chunk_size': avg_chunk_size,
                    'chunks_created': chunking.get('chunks_created', 0),
                    'timestamp': time.time()
                }
            else:
                return {
                    'status': 'unknown',
                    'message': 'No data available for quality assessment',
                    'timestamp': time.time()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _check_environment_health(self) -> dict:
        """Check environment health"""
        try:
            health_score = 100
            issues = []
            
            # Check Chroma API key
            chroma_config = self.env_loader.get_chroma_config()
            if not chroma_config['api_key']:
                health_score -= 50
                issues.append('CHROMA_API_KEY not set')
            
            # Check directories
            required_dirs = ['logs', 'reports', 'temp']
            for directory in required_dirs:
                if not Path(directory).exists():
                    health_score -= 10
                    issues.append(f'Directory {directory} not found')
            
            # Check disk space (simplified)
            disk_usage = self._get_disk_usage()
            if disk_usage > 90:
                health_score -= 20
                issues.append('High disk usage')
            
            status = 'excellent' if health_score >= 90 else 'good' if health_score >= 70 else 'fair' if health_score >= 50 else 'poor'
            
            return {
                'status': status,
                'health_score': health_score,
                'issues': issues,
                'disk_usage': disk_usage,
                'chroma_configured': bool(chroma_config['api_key']),
                'directories_ok': all(Path(d).exists() for d in required_dirs),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage (simplified)"""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            return (used / total) * 100
        except:
            return 0.0
    
    async def run_comprehensive_health_check(self) -> dict:
        """Run comprehensive health check"""
        print("Running comprehensive health check")
        print("=" * 50)
        
        components_to_check = ['chroma_cloud', 'pipeline', 'data_quality', 'environment']
        results = {}
        
        for component in components_to_check:
            result = await self.check_component_health(component)
            results[component] = result
        
        # Calculate overall health
        healthy_components = sum(1 for r in results.values() if r.get('status') in ['healthy', 'excellent', 'good'])
        overall_health_score = (healthy_components / len(components_to_check)) * 100
        
        if overall_health_score >= 80:
            overall_status = 'healthy'
        elif overall_health_score >= 60:
            overall_status = 'warning'
        else:
            overall_status = 'unhealthy'
        
        comprehensive_result = {
            'overall_status': overall_status,
            'overall_health_score': overall_health_score,
            'components': results,
            'check_duration': time.time() - self.start_time,
            'timestamp': time.time()
        }
        
        # Print summary
        print(f"\nOverall Health Status: {overall_status.upper()}")
        print(f"Overall Health Score: {overall_health_score:.1f}%")
        print(f"Healthy Components: {healthy_components}/{len(components_to_check)}")
        
        for component, result in results.items():
            status = result.get('status', 'unknown')
            print(f"  {component}: {status.upper()}")
        
        return comprehensive_result


async def main():
    """Main health check"""
    parser = argparse.ArgumentParser(description='Check pipeline health')
    parser.add_argument('--component', type=str, help='Specific component to check')
    parser.add_argument('--report', action='store_true', help='Generate health report')
    parser.add_argument('--output', type=str, help='Output file for report')
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    
    try:
        if args.component:
            # Check specific component
            result = await checker.check_component_health(args.component)
            print(f"\nComponent: {args.component}")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Details: {json.dumps(result, indent=2, default=str)}")
        else:
            # Run comprehensive check
            result = await checker.run_comprehensive_health_check()
            
            if args.report:
                # Save report
                output_file = args.output or f"reports/health_report_{datetime.now().strftime('%Y-%m-%d')}.json"
                Path('reports').mkdir(exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"\nHealth report saved to: {output_file}")
        
        # Exit with appropriate code
        overall_status = result.get('overall_status', result.get('status', 'unknown'))
        if overall_status in ['healthy', 'excellent', 'good']:
            print("Health check passed")
            sys.exit(0)
        elif overall_status == 'warning':
            print("Health check passed with warnings")
            sys.exit(1)
        else:
            print("Health check failed")
            sys.exit(2)
            
    except Exception as e:
        print(f"Health check failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
