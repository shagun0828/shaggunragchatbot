#!/usr/bin/env python3
"""
Phase 5-6 RAG Application Main Entry Point
FastAPI application with REST API, WebSocket, GraphQL, and advanced features
"""

import asyncio
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from api.rag_endpoints import router as rag_router
from api.chat_endpoints import router as chat_router
from api.search_endpoints import router as search_router
from api.monitoring_endpoints import router as monitoring_router
from websocket.websocket_manager import WebSocketManager
from graphql.graphql_app import graphql_app
from monitoring.metrics import MetricsCollector
from integration.chroma_client import ChromaClient
from integration.llm_client import LLMClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logging.info("Starting Phase 5-6 RAG Application")
    
    # Initialize components
    app.state.chroma_client = ChromaClient()
    app.state.llm_client = LLMClient()
    app.state.websocket_manager = WebSocketManager()
    app.state.metrics = MetricsCollector()
    
    # Initialize connections
    await app.state.chroma_client.initialize()
    await app.state.llm_client.initialize()
    
    logging.info("Application startup completed")
    
    yield
    
    # Shutdown
    logging.info("Shutting down Phase 5-6 RAG Application")
    
    # Cleanup
    await app.state.chroma_client.close()
    await app.state.llm_client.close()
    await app.state.websocket_manager.disconnect_all()
    
    logging.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Phase 5-6 RAG Application",
    description="Advanced RAG system with multi-modal capabilities, real-time processing, and personalization",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(search_router, prefix="/api/v1/search", tags=["Search"])
app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])

# GraphQL endpoint
app.add_route("/graphql", graphql_app)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time communication"""
    await app.state.websocket_manager.connect(websocket)

# Static files for UI
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 5-6 RAG Application</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .feature { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 5-6 RAG Application</h1>
            <p>Advanced RAG system with multi-modal capabilities, real-time processing, and personalization</p>
            
            <div class="feature">
                <h2>Chat Interface</h2>
                <p>Interactive chat with AI assistant powered by RAG</p>
                <button class="btn" onclick="window.open('/static/chat.html', '_blank')">Open Chat</button>
            </div>
            
            <div class="feature">
                <h2>Search Interface</h2>
                <p>AI-enhanced search with semantic capabilities</p>
                <button class="btn" onclick="window.open('/static/search.html', '_blank')">Open Search</button>
            </div>
            
            <div class="feature">
                <h2>API Documentation</h2>
                <p>Interactive API documentation</p>
                <button class="btn" onclick="window.open('/docs', '_blank')">Open Docs</button>
            </div>
            
            <div class="feature">
                <h2>GraphQL Playground</h2>
                <p>Flexible query interface</p>
                <button class="btn" onclick="window.open('/graphql', '_blank')">Open GraphQL</button>
            </div>
            
            <div class="feature">
                <h2>Monitoring Dashboard</h2>
                <p>System metrics and analytics</p>
                <button class="btn" onclick="window.open('/static/dashboard.html', '_blank')">Open Dashboard</button>
            </div>
        </div>
    </body>
    </html>
    """

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "chroma_client": await app.state.chroma_client.health_check(),
            "llm_client": await app.state.llm_client.health_check(),
            "websocket_manager": app.state.websocket_manager.get_status(),
            "metrics": app.state.metrics.get_status()
        }
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return app.state.metrics.get_all_metrics()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
