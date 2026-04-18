#!/usr/bin/env python3
"""
Pipeline Report Generator
Generates comprehensive reports for daily ingest pipeline
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))


class PipelineReportGenerator:
    """Generates comprehensive pipeline reports"""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_report(self, date: str, output_file: str = None) -> dict:
        """Generate comprehensive pipeline report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Generating pipeline report for {date}")
        print("=" * 50)
        
        # Load pipeline results
        pipeline_file = f"reports/daily_ingest_{date}.json"
        
        if not Path(pipeline_file).exists():
            print(f"ERROR: Pipeline results file not found: {pipeline_file}")
            return {'status': 'failed', 'reason': 'pipeline_results_not_found'}
        
        with open(pipeline_file, 'r') as f:
            pipeline_results = json.load(f)
        
        # Load verification results
        verification_file = f"reports/chroma_verification_{date}.json"
        verification_results = {}
        
        if Path(verification_file).exists():
            with open(verification_file, 'r') as f:
                verification_results = json.load(f)
        
        # Generate comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_date': date,
                'report_type': 'daily_ingest_pipeline',
                'version': '1.0'
            },
            'pipeline_execution': pipeline_results,
            'chroma_verification': verification_results,
            'summary': self._generate_summary(pipeline_results, verification_results),
            'recommendations': self._generate_recommendations(pipeline_results, verification_results),
            'quality_metrics': self._calculate_quality_metrics(pipeline_results),
            'performance_metrics': self._calculate_performance_metrics(pipeline_results)
        }
        
        # Save report
        if output_file is None:
            output_file = f"reports/pipeline_report_{date}.json"
        
        Path('reports').mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_file}")
        
        # Generate human-readable summary
        self._print_summary(report)
        
        return report
    
    def _generate_summary(self, pipeline_results: dict, verification_results: dict) -> dict:
        """Generate executive summary"""
        summary = {
            'overall_status': pipeline_results.get('status', 'unknown'),
            'pipeline_efficiency': 'good',
            'data_quality': 'good',
            'chroma_cloud_status': 'connected',
            'key_achievements': [],
            'issues_identified': []
        }
        
        # Add achievements
        if pipeline_results.get('status') == 'completed':
            summary['key_achievements'].append("Pipeline completed successfully")
            
            scraping = pipeline_results.get('scraping_results', {})
            if scraping.get('urls_scraped', 0) > 0:
                summary['key_achievements'].append(f"Scraped {scraping['urls_scraped']} URLs successfully")
            
            chunking = pipeline_results.get('chunking_results', {})
            if chunking.get('chunks_created', 0) > 0:
                summary['key_achievements'].append(f"Created {chunking['chunks_created']} chunks")
            
            upload = pipeline_results.get('upload_results', {}).get('cloud_results', {})
            if upload.get('total_uploads', 0) > 0:
                summary['key_achievements'].append(f"Uploaded {upload['total_uploads']} documents to Chroma Cloud")
        
        # Add issues
        scraping_errors = pipeline_results.get('scraping_results', {}).get('scraping_errors', 0)
        if scraping_errors > 0:
            summary['issues_identified'].append(f"Scraping errors: {scraping_errors}")
        
        if verification_results.get('overall', {}).get('status') != 'completed':
            summary['issues_identified'].append("Chroma Cloud verification failed")
        
        # Determine overall quality
        if scraping_errors > 5:
            summary['data_quality'] = 'poor'
        elif scraping_errors > 0:
            summary['data_quality'] = 'fair'
        
        return summary
    
    def _generate_recommendations(self, pipeline_results: dict, verification_results: dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze scraping performance
        scraping = pipeline_results.get('scraping_results', {})
        scraping_time = scraping.get('scraping_time', 0)
        urls_scraped = scraping.get('urls_scraped', 0)
        
        if scraping_time > 60:
            recommendations.append("Consider optimizing scraping performance - took too long")
        
        if urls_scraped < 20:
            recommendations.append("Review scraping sources - low success rate")
        
        # Analyze chunking performance
        chunking = pipeline_results.get('chunking_results', {})
        avg_chunk_size = chunking.get('avg_chunk_size', 0)
        
        if avg_chunk_size < 100:
            recommendations.append("Chunks are too small - consider increasing chunk size target")
        elif avg_chunk_size > 1500:
            recommendations.append("Chunks are too large - consider decreasing chunk size target")
        
        # Analyze upload performance
        upload = pipeline_results.get('upload_results', {}).get('cloud_results', {})
        upload_time = upload.get('processing_time', 0)
        
        if upload_time > 30:
            recommendations.append("Consider optimizing upload performance - took too long")
        
        # Analyze verification results
        if verification_results.get('overall', {}).get('status') != 'completed':
            recommendations.append("Investigate Chroma Cloud upload issues")
        
        if not recommendations:
            recommendations.append("Pipeline is performing optimally - no immediate actions needed")
        
        return recommendations
    
    def _calculate_quality_metrics(self, pipeline_results: dict) -> dict:
        """Calculate quality metrics"""
        metrics = {}
        
        # Scraping quality
        scraping = pipeline_results.get('scraping_results', {})
        urls_attempted = scraping.get('urls_attempted', 1)
        urls_scraped = scraping.get('urls_scraped', 0)
        
        metrics['scraping_success_rate'] = (urls_scraped / urls_attempted) * 100 if urls_attempted > 0 else 0
        
        # Chunk quality
        chunking = pipeline_results.get('chunking_results', {})
        chunks_created = chunking.get('chunks_created', 0)
        avg_chunk_size = chunking.get('avg_chunk_size', 0)
        
        metrics['chunk_quality_score'] = self._calculate_chunk_quality_score(avg_chunk_size)
        
        # Upload quality
        upload = pipeline_results.get('upload_results', {}).get('cloud_results', {})
        total_uploads = upload.get('total_uploads', 0)
        collections_updated = len(upload.get('upload_results', {}))
        
        metrics['upload_success_rate'] = 100 if total_uploads > 0 else 0
        
        # Overall quality score
        metrics['overall_quality_score'] = (
            metrics['scraping_success_rate'] * 0.4 +
            metrics['chunk_quality_score'] * 0.3 +
            metrics['upload_success_rate'] * 0.3
        )
        
        return metrics
    
    def _calculate_chunk_quality_score(self, avg_chunk_size: float) -> float:
        """Calculate chunk quality score based on size"""
        # Optimal chunk size is 800-1200 characters
        if 800 <= avg_chunk_size <= 1200:
            return 100
        elif 600 <= avg_chunk_size < 800 or 1200 < avg_chunk_size <= 1500:
            return 80
        elif 400 <= avg_chunk_size < 600 or 1500 < avg_chunk_size <= 2000:
            return 60
        else:
            return 40
    
    def _calculate_performance_metrics(self, pipeline_results: dict) -> dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # Throughput metrics
        scraping = pipeline_results.get('scraping_results', {})
        chunking = pipeline_results.get('chunking_results', {})
        upload = pipeline_results.get('upload_results', {})
        
        metrics['scraping_throughput'] = scraping.get('urls_scraped', 0) / max(scraping.get('scraping_time', 0.001), 0.001)
        metrics['chunking_throughput'] = chunking.get('chunks_created', 0) / max(chunking.get('chunking_time', 0.001), 0.001)
        metrics['upload_throughput'] = upload.get('cloud_results', {}).get('total_uploads', 0) / max(upload.get('processing_time', 0.001), 0.001)
        
        # Time distribution
        total_time = pipeline_results.get('metrics', {}).get('total_duration', 0)
        if total_time > 0:
            metrics['scraping_time_percentage'] = (scraping.get('scraping_time', 0) / total_time) * 100
            metrics['chunking_time_percentage'] = (chunking.get('chunking_time', 0) / total_time) * 100
            metrics['upload_time_percentage'] = (upload.get('processing_time', 0) / total_time) * 100
            metrics['overall_throughput'] = scraping.get('urls_scraped', 0) / total_time
        else:
            metrics['scraping_time_percentage'] = 0
            metrics['chunking_time_percentage'] = 0
            metrics['upload_time_percentage'] = 0
            metrics['overall_throughput'] = 0
        
        return metrics
    
    def _print_summary(self, report: dict):
        """Print human-readable summary"""
        print("\n" + "=" * 50)
        print("PIPELINE REPORT SUMMARY")
        print("=" * 50)
        
        summary = report['summary']
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Pipeline Efficiency: {summary['pipeline_efficiency'].upper()}")
        print(f"Data Quality: {summary['data_quality'].upper()}")
        print(f"Chroma Cloud Status: {summary['chroma_cloud_status'].upper()}")
        
        print("\nKey Achievements:")
        for achievement in summary['key_achievements']:
            print(f"  â {achievement}")
        
        if summary['issues_identified']:
            print("\nIssues Identified:")
            for issue in summary['issues_identified']:
                print(f"  â {issue}")
        
        print("\nQuality Metrics:")
        quality = report['quality_metrics']
        print(f"  Scraping Success Rate: {quality['scraping_success_rate']:.1f}%")
        print(f"  Chunk Quality Score: {quality['chunk_quality_score']:.1f}")
        print(f"  Upload Success Rate: {quality['upload_success_rate']:.1f}%")
        print(f"  Overall Quality Score: {quality['overall_quality_score']:.1f}")
        
        print("\nPerformance Metrics:")
        performance = report['performance_metrics']
        print(f"  Scraping Throughput: {performance['scraping_throughput']:.2f} URLs/s")
        print(f"  Chunking Throughput: {performance['chunking_throughput']:.2f} chunks/s")
        print(f"  Upload Throughput: {performance['upload_throughput']:.2f} docs/s")
        print(f"  Overall Throughput: {performance['overall_throughput']:.2f} URLs/s")
        
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 50)


async def main():
    """Main report generator"""
    parser = argparse.ArgumentParser(description='Generate pipeline report')
    parser.add_argument('--date', type=str, help='Date to generate report for (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    generator = PipelineReportGenerator()
    
    try:
        report = generator.generate_report(args.date, args.output)
        
        # Exit with appropriate code
        if report['summary']['overall_status'] == 'completed':
            print("Report generated successfully")
            sys.exit(0)
        else:
            print("Report generated with issues")
            sys.exit(1)
            
    except Exception as e:
        print(f"Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
