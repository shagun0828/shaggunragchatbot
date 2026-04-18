#!/usr/bin/env python3
"""
Chroma Cloud Upload Verification Script
Verifies that data was successfully uploaded to Chroma Cloud
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from simple_chroma_cloud import ChromaCloudManager
from utils.env_loader import env_loader


class ChromaUploadVerifier:
    """Verifies Chroma Cloud uploads"""
    
    def __init__(self):
        self.chroma_manager = ChromaCloudManager()
    
    async def verify_upload(self, date: str = None) -> dict:
        """Verify upload for specific date"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Verifying Chroma Cloud upload for {date}")
        print("=" * 50)
        
        # Check connection
        health = await self.chroma_manager.health_check()
        print(f"Chroma Cloud Status: {health['status']}")
        
        if health['status'] != 'healthy':
            print(f"ERROR: Chroma Cloud connection failed: {health.get('error', 'Unknown error')}")
            return {'status': 'failed', 'reason': 'connection_failed'}
        
        # Check collections
        collections_to_check = ['mutual_funds_v1', 'financial_news_v1']
        verification_results = {}
        
        for collection_name in collections_to_check:
            print(f"\nChecking collection: {collection_name}")
            
            # Get collection stats
            stats = await self.chroma_manager.get_collection_stats(collection_name)
            
            if stats:
                print(f"  Documents: {stats['document_count']}")
                print(f"  Created: {datetime.fromtimestamp(stats['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Verify recent uploads
                recent_docs = await self._verify_recent_documents(collection_name, date)
                verification_results[collection_name] = {
                    'status': 'verified',
                    'document_count': stats['document_count'],
                    'recent_documents': recent_docs,
                    'created_at': stats['created_at']
                }
            else:
                print(f"  ERROR: Collection not found")
                verification_results[collection_name] = {
                    'status': 'not_found',
                    'document_count': 0
                }
        
        # Get overall metrics
        metrics = self.chroma_manager.get_metrics()
        print(f"\nOverall Chroma Cloud Metrics:")
        print(f"  Total Documents Uploaded: {metrics['current_metrics']['documents_uploaded']}")
        print(f"  Collections Created: {metrics['current_metrics']['collections_created']}")
        print(f"  Upload Time: {metrics['current_metrics']['upload_time']:.2f}s")
        print(f"  Avg Upload Speed: {metrics['current_metrics']['avg_upload_speed']:.2f} docs/s")
        
        # Test search functionality
        print(f"\nTesting search functionality...")
        search_results = await self._test_search_functionality()
        
        verification_results['overall'] = {
            'status': 'completed',
            'health_status': health['status'],
            'total_documents': metrics['current_metrics']['documents_uploaded'],
            'collections': list(metrics['collections']),
            'search_test': search_results,
            'verification_date': date
        }
        
        return verification_results
    
    async def _verify_recent_documents(self, collection_name: str, date: str) -> int:
        """Verify recent documents in collection"""
        try:
            # Generate a test query
            import numpy as np
            query_embedding = np.random.rand(768 if 'mutual_funds' in collection_name else 384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search for recent documents
            results = await self.chroma_manager.search_embeddings(
                query_embedding, collection_name, top_k=10
            )
            
            # Count documents from today
            today_count = 0
            for result in results:
                metadata = result.get('metadata', {})
                run_date = metadata.get('run_date', '')
                if run_date == date:
                    today_count += 1
            
            print(f"  Recent documents from {date}: {today_count}")
            return today_count
            
        except Exception as e:
            print(f"  ERROR: Failed to verify recent documents: {e}")
            return 0
    
    async def _test_search_functionality(self) -> dict:
        """Test search functionality"""
        try:
            import numpy as np
            
            # Test mutual funds collection
            query_embedding = np.random.rand(768)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            results = await self.chroma_manager.search_embeddings(
                query_embedding, 'mutual_funds_v1', top_k=3
            )
            
            return {
                'status': 'success',
                'results_found': len(results),
                'sample_distances': [r['distance'] for r in results[:3]]
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }


async def main():
    """Main verification script"""
    parser = argparse.ArgumentParser(description='Verify Chroma Cloud upload')
    parser.add_argument('--date', type=str, help='Date to verify (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    verifier = ChromaUploadVerifier()
    
    try:
        results = await verifier.verify_upload(args.date)
        
        # Save verification results
        output_file = f"reports/chroma_verification_{args.date or datetime.now().strftime('%Y-%m-%d')}.json"
        Path('reports').mkdir(exist_ok=True)
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nVerification results saved to: {output_file}")
        
        # Exit with appropriate code
        if results['overall']['status'] == 'completed':
            print("Verification completed successfully")
            sys.exit(0)
        else:
            print("Verification failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
