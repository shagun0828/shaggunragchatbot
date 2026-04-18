#!/usr/bin/env python3
"""
Notification System for Pipeline
Sends notifications for pipeline status
"""

import argparse
import asyncio
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.env_loader import EnvLoader


class NotificationSender:
    """Sends notifications for pipeline status"""
    
    def __init__(self):
        self.env_loader = EnvLoader()
        self.webhook_url = self.env_loader.get_str('NOTIFICATION_WEBHOOK')
    
    async def send_notification(self, status: str, message: str, details: dict = None) -> bool:
        """Send notification"""
        print(f"Sending notification: {status.upper()} - {message}")
        
        if not self.webhook_url:
            print("No webhook URL configured, skipping notification")
            return False
        
        try:
            import aiohttp
            
            # Prepare notification payload
            payload = {
                'status': status,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'service': 'daily_ingest_pipeline',
                'run_date': os.getenv('RUN_DATE', datetime.now().strftime('%Y-%m-%d')),
                'run_id': os.getenv('RUN_ID', 'unknown'),
                'details': details or {}
            }
            
            # Send notification
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print("Notification sent successfully")
                        return True
                    else:
                        print(f"Failed to send notification: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            print(f"Error sending notification: {e}")
            return False
    
    async def send_success_notification(self, message: str, details: dict = None) -> bool:
        """Send success notification"""
        return await self.send_notification('success', message, details)
    
    async def send_failure_notification(self, message: str, details: dict = None) -> bool:
        """Send failure notification"""
        return await self.send_notification('failure', message, details)
    
    async def send_warning_notification(self, message: str, details: dict = None) -> bool:
        """Send warning notification"""
        return await self.send_notification('warning', message, details)


async def main():
    """Main notification sender"""
    parser = argparse.ArgumentParser(description='Send pipeline notification')
    parser.add_argument('--status', type=str, required=True, choices=['success', 'failure', 'warning'])
    parser.add_argument('--message', type=str, required=True, help='Notification message')
    parser.add_argument('--details', type=str, help='JSON string with additional details')
    
    args = parser.parse_args()
    
    sender = NotificationSender()
    
    try:
        # Parse details if provided
        details = None
        if args.details:
            details = json.loads(args.details)
        
        # Send notification
        success = await sender.send_notification(args.status, args.message, details)
        
        if success:
            print("Notification sent successfully")
            sys.exit(0)
        else:
            print("Failed to send notification")
            sys.exit(1)
            
    except Exception as e:
        print(f"Notification sending failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
