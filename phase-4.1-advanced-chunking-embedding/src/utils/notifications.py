"""
Notification manager for Phase 4.1
Handles notifications for advanced chunking and embedding processes
"""

import asyncio
import logging
import os
from typing import Optional
import aiohttp
import json
from datetime import datetime


class NotificationManager:
    """Manages notifications for advanced processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.webhook_url = os.getenv('NOTIFICATION_WEBHOOK')
        self.email_enabled = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
        
    async def send_success_notification(self, message: str) -> None:
        """Send success notification"""
        await self._send_notification("SUCCESS", message)
    
    async def send_failure_notification(self, message: str) -> None:
        """Send failure notification"""
        await self._send_notification("FAILURE", message)
    
    async def send_warning_notification(self, message: str) -> None:
        """Send warning notification"""
        await self._send_notification("WARNING", message)
    
    async def send_quality_alert(self, quality_score: float, issues: list) -> None:
        """Send quality alert notification"""
        message = (
            f"Quality Alert: Score {quality_score:.3f}\n"
            f"Issues detected: {len(issues)}\n"
            f"Main issues: {', '.join(issues[:3])}"
        )
        await self._send_notification("QUALITY_ALERT", message)
    
    async def send_processing_summary(self, stats: dict) -> None:
        """Send processing summary notification"""
        message = (
            f"Advanced Processing Summary:\n"
            f"- Funds processed: {stats.get('funds_processed', 0)}\n"
            f"- Chunks created: {stats.get('chunks_created', 0)}\n"
            f"- Embeddings generated: {stats.get('embeddings_generated', 0)}\n"
            f"- Quality issues fixed: {stats.get('quality_issues_fixed', 0)}\n"
            f"- Processing time: {stats.get('processing_time', 0):.2f}s"
        )
        await self.send_success_notification(message)
    
    async def _send_notification(self, level: str, message: str) -> None:
        """Send notification through available channels"""
        
        # Send webhook notification if configured
        if self.webhook_url:
            await self._send_webhook_notification(level, message)
        
        # Send email notification if configured
        if self.email_enabled:
            await self._send_email_notification(level, message)
        
        # Log the notification
        log_message = f"[{level}] {message}"
        if level == "SUCCESS":
            self.logger.info(log_message)
        elif level == "FAILURE":
            self.logger.error(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "QUALITY_ALERT":
            self.logger.warning(log_message)
    
    async def _send_webhook_notification(self, level: str, message: str) -> None:
        """Send webhook notification"""
        try:
            payload = {
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "advanced_chunking_embedding",
                "phase": "4.1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.logger.info("Webhook notification sent successfully")
                    else:
                        self.logger.warning(f"Webhook notification failed with status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {str(e)}")
    
    async def _send_email_notification(self, level: str, message: str) -> None:
        """Send email notification (placeholder implementation)"""
        # This would be actual email implementation
        # For now, we'll just log it
        self.logger.info(f"Email notification would be sent: [{level}] {message}")
