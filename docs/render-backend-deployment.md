# Render Backend Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites
- GitHub repository: `https://github.com/shagun0828/shaggunragchatbot.git`
- Render account (free tier available)
- Environment variables from `.env` file

---

## Step 1: Set Up Render Account

### 1.1 Create Render Account
1. Go to [https://render.com](https://render.com)
2. Click "Sign Up"
3. Sign up with GitHub (recommended)
4. Authorize Render to access your GitHub repositories

### 1.2 Connect Repository
1. After signing in, click "New +"
2. Select "Web Service"
3. Choose "Connect a repository"
4. Select `shagun0828/shaggunragchatbot`
5. Click "Connect"

---

## Step 2: Configure Web Service

### 2.1 Basic Configuration
1. **Name**: `rag-backend`
2. **Environment**: `Python`
3. **Region**: Select nearest region (recommended: Oregon)
4. **Branch**: `main`
5. **Root Directory**: `phase-5-6-rag-application`

### 2.2 Build Configuration
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`

### 2.3 Advanced Configuration
1. **Health Check Path**: `/api/v1/monitoring/health`
2. **Auto-Deploy**: Enable (checked)
3. **Instance Type**: Free (to start, can upgrade later)

---

## Step 3: Environment Variables

### 3.1 Add Environment Variables
Navigate to the "Environment" tab and add these variables:

#### Chroma Cloud Configuration
```
CHROMA_API_KEY=ck-DKi8GLXjuW48HS2ctLNW8TxdmvKzMVLgfk1XuNoW9kXM
CHROMA_TENANT=default
CHROMA_DATABASE=mutual-funds-db
ENABLE_CHROMA_CLOUD=true
```

#### LLM Configuration
```
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002
```

#### Application Configuration
```
PYTHON_VERSION=3.11
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
```

#### Monitoring Configuration
```
ENABLE_METRICS=true
METRICS_PORT=9090
```

#### WebSocket Configuration
```
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_CONNECTION_TIMEOUT=3600
```

#### Personalization Configuration
```
ENABLE_PERSONALIZATION=true
USER_PROFILE_TTL=86400
```

#### Rate Limiting
```
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

---

## Step 4: Deploy the Backend

### 4.1 Initial Deployment
1. Click "Create Web Service"
2. Wait for the build to complete (2-5 minutes)
3. Monitor the build logs for any errors

### 4.2 Verify Deployment
Once deployed, you should see:
- **Service URL**: `https://rag-backend.onrender.com`
- **Status**: "Live"
- **Health Check**: Passing

---

## Step 5: Test the Deployment

### 5.1 Health Check
```bash
curl https://rag-backend.onrender.com/api/v1/monitoring/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-04-18T19:30:00Z",
  "version": "2.0.0",
  "services": {
    "chroma_cloud": "connected",
    "database": "operational",
    "llm": "configured"
  }
}
```

### 5.2 Test API Endpoints
```bash
# Test root endpoint
curl https://rag-backend.onrender.com/

# Test API documentation
curl https://rag-backend.onrender.com/docs

# Test GraphQL endpoint
curl -X POST https://rag-backend.onrender.com/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __schema { types { name } } }"}'
```

### 5.3 Test WebSocket Connection
```javascript
// Test WebSocket connection in browser console
const ws = new WebSocket('wss://rag-backend.onrender.com/ws');
ws.onopen = () => console.log('WebSocket connected');
ws.onmessage = (event) => console.log('Received:', event.data);
```

---

## Step 6: Configure Custom Settings

### 6.1 CORS Configuration
The backend is already configured to allow requests from:
- `https://rag-frontend.vercel.app`

### 6.2 Custom Domain (Optional)
1. Go to "Custom Domains" tab
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for SSL certificate issuance

### 6.3 Scaling Configuration (Optional)
1. Go to "Settings" tab
2. Adjust instance type based on needs:
   - **Free**: 256MB RAM, 0.1 CPU
   - **Starter**: 512MB RAM, 0.25 CPU ($7/month)
   - **Standard**: 1GB RAM, 0.5 CPU ($25/month)

---

## Step 7: Monitoring and Logs

### 7.1 View Logs
1. Go to "Logs" tab
2. Monitor real-time logs
3. Check for any errors or warnings

### 7.2 Metrics Dashboard
1. Go to "Metrics" tab
2. Monitor:
   - Response times
   - Request rates
   - Error rates
   - Resource usage

### 7.3 Set Up Alerts (Optional)
1. Go to "Alerts" tab
2. Configure alerts for:
   - High error rates
   - Slow response times
   - Service downtime

---

## Troubleshooting

### Common Issues

#### Build Failures
- **Issue**: Requirements.txt not found
- **Solution**: Ensure file is in `phase-5-6-rag-application/` directory

#### Runtime Errors
- **Issue**: Port binding errors
- **Solution**: Ensure `$PORT` environment variable is used

#### Health Check Failures
- **Issue**: Health check endpoint not responding
- **Solution**: Verify `/api/v1/monitoring/health` endpoint exists

#### Environment Variable Issues
- **Issue**: API keys not working
- **Solution**: Double-check environment variable names and values

#### CORS Issues
- **Issue**: Frontend cannot access backend
- **Solution**: Verify CORS configuration includes frontend domain

### Debug Commands

```bash
# Check service status
curl https://rag-backend.onrender.com/api/v1/monitoring/health

# Test with verbose output
curl -v https://rag-backend.onrender.com/api/v1/monitoring/health

# Check logs in Render dashboard
# Navigate to your service > Logs tab
```

---

## Post-Deployment Checklist

### Verification
- [ ] Service is live and accessible
- [ ] Health check endpoint responding
- [ ] API endpoints working correctly
- [ ] WebSocket connections functional
- [ ] Environment variables properly configured
- [ ] CORS settings allowing frontend access
- [ ] Logs showing no critical errors
- [ ] Metrics dashboard operational

### Security
- [ ] API keys are properly secured
- [ ] HTTPS is enforced
- [ ] Rate limiting is configured
- [ ] No sensitive data in logs
- [ ] CORS properly configured

### Performance
- [ ] Response times under 2 seconds
- [ ] Memory usage within limits
- [ ] Error rate below 1%
- [ ] Auto-scaling configured if needed

---

## Next Steps

### 1. Frontend Deployment
- Deploy frontend to Vercel
- Update frontend environment variables
- Test frontend-backend integration

### 2. GitHub Actions Setup
- Configure GitHub Secrets
- Test automated scheduler
- Monitor daily executions

### 3. Monitoring Setup
- Configure alerting
- Set up log aggregation
- Implement performance monitoring

### 4. Documentation Update
- Update API documentation
- Document deployment process
- Create user guides

---

## Support Resources

### Render Documentation
- [Render Docs](https://render.com/docs)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Web Services](https://render.com/docs/web-services)

### Common Issues
- [Troubleshooting Guide](https://render.com/docs/troubleshooting)
- [Best Practices](https://render.com/docs/best-practices)

### Community Support
- [Render Community](https://community.render.com)
- [GitHub Discussions](https://github.com/renderinc/render/discussions)

---

## Cost Information

### Free Tier Limits
- **RAM**: 256MB
- **CPU**: 0.1 vCPU
- **Bandwidth**: 100GB/month
- **Build Time**: 750 minutes/month

### Upgrade Options
- **Starter**: $7/month (512MB RAM, 0.25 CPU)
- **Standard**: $25/month (1GB RAM, 0.5 CPU)
- **Pro**: $50/month (2GB RAM, 1 CPU)

### Cost Optimization Tips
- Monitor resource usage
- Optimize code for memory efficiency
- Use caching where possible
- Scale based on actual needs
