"""
Notification manager for Phase 4.2
Handles notifications for real-time optimization processes
"""

import asyncio
import logging
import os
from typing import Optional
import aiohttp
import json
from datetime import datetime


class NotificationManager:
    """Manages notifications for real-time processing"""
    
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
    
    async def _send_webhook_notification(self, level: str, message: str) -> None:
        """Send webhook notification"""
        try:
            payload = {
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "realtime_optimization",
                "phase": "4.2"
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
