# Render Backend Deployment Checklist

## Pre-Deployment Checklist
- [ ] Render account created and signed in
- [ ] GitHub repository connected to Render
- [ ] Environment variables ready from .env file
- [ ] render.yaml file in phase-5-6-rag-application directory
- [ ] requirements.txt file exists and is complete

## Deployment Steps
- [ ] Go to https://render.com and sign in
- [ ] Click "New +" > "Web Service"
- [ ] Connect to shagun0828/shaggunragchatbot repository
- [ ] Configure basic settings (name, environment, etc.)
- [ ] Set root directory to "phase-5-6-rag-application"
- [ ] Configure build and start commands
- [ ] Add all environment variables
- [ ] Click "Create Web Service"
- [ ] Wait for deployment to complete
- [ ] Verify service is live

## Post-Deployment Verification
- [ ] Service URL accessible: https://rag-backend.onrender.com
- [ ] Health check endpoint working: /api/v1/monitoring/health
- [ ] API documentation accessible: /docs
- [ ] No errors in build logs
- [ ] Environment variables properly loaded
- [ ] CORS configuration working

## Testing Checklist
- [ ] Health check endpoint returns 200 OK
- [ ] API endpoints responding correctly
- [ ] WebSocket connection functional
- [ ] GraphQL endpoint working
- [ ] Error handling working properly

## Environment Variables to Add
```
CHROMA_API_KEY=ck-DKi8GLXjuW48HS2ctLNW8TxdmvKzMVLgfk1XuNoW9kXM
CHROMA_TENANT=default
CHROMA_DATABASE=mutual-funds-db
ENABLE_CHROMA_CLOUD=true
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
PYTHON_VERSION=3.11
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
ENABLE_METRICS=true
METRICS_PORT=9090
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_CONNECTION_TIMEOUT=3600
ENABLE_PERSONALIZATION=true
USER_PROFILE_TTL=86400
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

## URLs to Test After Deployment
- Health Check: https://rag-backend.onrender.com/api/v1/monitoring/health
- API Docs: https://rag-backend.onrender.com/docs
- Root: https://rag-backend.onrender.com/
- GraphQL: https://rag-backend.onrender.com/graphql
- WebSocket: wss://rag-backend.onrender.com/ws
