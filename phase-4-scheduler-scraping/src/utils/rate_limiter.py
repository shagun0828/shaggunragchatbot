"""
Rate limiter for web scraping
Controls request frequency to avoid being blocked
"""

import asyncio
import time
from typing import Optional
import logging


class RateLimiter:
    """Rate limiter for web requests"""
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def acquire(self) -> None:
        """Acquire a token from the rate limiter"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            tokens_to_add = elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                self.logger.debug(f"Token acquired. Tokens remaining: {self.tokens}")
            else:
                # Calculate wait time
                wait_time = (1 - self.tokens) / self.requests_per_second
                self.logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                await self.acquire()  # Retry after waiting
    
    async def wait(self) -> None:
        """Convenience method to wait for rate limit"""
        await self.acquire()


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket"""
        async with self.lock:
            await self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> None:
        """Wait until enough tokens are available"""
        while not await self.consume(tokens):
            await asyncio.sleep(0.1)
    
    async def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
