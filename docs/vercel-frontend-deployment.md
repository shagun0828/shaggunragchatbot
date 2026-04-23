# Vercel Frontend Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites
- GitHub repository: `https://github.com/shagun0828/shaggunragchatbot.git`
- Vercel account (free tier available)
- Backend deployed on Render
- Next.js frontend code in `phase-5-6-rag-application/frontend`

---

## Step 1: Set Up Vercel Account

### 1.1 Create Vercel Account
1. Go to [https://vercel.com](https://vercel.com)
2. Click "Sign Up"
3. Sign up with GitHub (recommended)
4. Authorize Vercel to access your GitHub repositories

### 1.2 Connect Repository
1. After signing in, click "New Project"
2. Select "Import Git Repository"
3. Choose `shagun0828/shaggunragchatbot`
4. Click "Import"

---

## Step 2: Configure Project Settings

### 2.1 Basic Configuration
1. **Project Name**: `rag-frontend`
2. **Framework Preset**: Next.js (should be auto-detected)
3. **Root Directory**: `phase-5-6-rag-application/frontend`
4. **Build Command**: `npm run build` (auto-detected)
5. **Output Directory**: `.next` (auto-detected)
6. **Install Command**: `npm install` (auto-detected)

### 2.2 Environment Variables
Navigate to the "Environment Variables" section and add these variables:

#### API Configuration
```
NEXT_PUBLIC_API_URL=https://rag-backend.onrender.com
NEXT_PUBLIC_WS_URL=wss://rag-backend.onrender.com/ws
NEXT_PUBLIC_GRAPHQL_URL=https://rag-backend.onrender.com/graphql
```

#### Application Configuration
```
NEXT_PUBLIC_APP_NAME=RAG System
NEXT_PUBLIC_APP_VERSION=2.0.0
NEXT_PUBLIC_ENVIRONMENT=production
```

---

## Step 3: Deploy the Frontend

### 3.1 Initial Deployment
1. Click "Deploy"
2. Wait for the build to complete (2-5 minutes)
3. Monitor the build logs for any errors

### 3.2 Verify Deployment
Once deployed, you should see:
- **Deployment URL**: `https://rag-frontend.vercel.app`
- **Status**: "Ready"
- **Build**: Successful

---

## Step 4: Configure Custom Settings

### 4.1 Domain Configuration (Optional)
1. Go to "Domains" tab
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for SSL certificate issuance

### 4.2 Build Optimization
1. Go to "Settings" > "Build & Development Settings"
2. Configure:
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

### 4.3 Environment Variables Management
1. Go to "Settings" > "Environment Variables"
2. Ensure all variables are properly configured
3. Test with different environments if needed

---

## Step 5: Test the Deployment

### 5.1 Basic Functionality
1. **Open**: `https://rag-frontend.vercel.app`
2. **Check**: Page loads correctly
3. **Verify**: Dark theme is applied
4. **Test**: Navigation between components

### 5.2 API Integration Tests
1. **Chat Interface**: Test connection to backend
2. **Search Interface**: Test search functionality
3. **Dashboard**: Test data loading
4. **WebSocket**: Test real-time connections

### 5.3 Cross-Browser Testing
1. **Chrome**: Full functionality
2. **Firefox**: Full functionality
3. **Safari**: Full functionality
4. **Mobile**: Responsive design

---

## Step 6: Monitor and Troubleshoot

### 6.1 View Logs
1. Go to "Logs" tab
2. Monitor real-time logs
3. Check for any errors or warnings
4. Filter by function or time

### 6.2 Analytics Dashboard
1. Go to "Analytics" tab
2. Monitor:
   - Page views
   - Unique visitors
   - Performance metrics
   - Error rates

### 6.3 Performance Optimization
1. Go to "Speed Insights" tab
2. Analyze:
   - Core Web Vitals
   - Page load times
   - Bundle size analysis
   - Optimization suggestions

---

## Step 7: Advanced Configuration

### 7.1 Edge Functions
1. Create API routes in `pages/api/`
2. Configure edge functions for performance
3. Test edge function responses

### 7.2 Caching Strategy
1. Configure cache headers
2. Set up ISR (Incremental Static Regeneration)
3. Optimize asset caching

### 7.3 Security Configuration
1. Configure CSP headers
2. Set up rate limiting
3. Implement security best practices

---

## Troubleshooting

### Common Issues

#### Build Failures
- **Issue**: Dependencies not found
- **Solution**: Check package.json and node_modules
- **Fix**: Ensure all dependencies are listed correctly

#### Runtime Errors
- **Issue**: API connection failures
- **Solution**: Check environment variables
- **Fix**: Verify backend URL is correct

#### WebSocket Issues
- **Issue**: WebSocket connection fails
- **Solution**: Check WebSocket URL configuration
- **Fix**: Ensure wss:// protocol is used

#### CORS Issues
- **Issue**: Frontend cannot access backend
- **Solution**: Check CORS configuration on backend
- **Fix**: Ensure frontend domain is allowed

#### Environment Variable Issues
- **Issue**: Variables not working
- **Solution**: Check variable names and values
- **Fix**: Ensure NEXT_PUBLIC_ prefix for client-side variables

### Debug Commands

```bash
# Test frontend locally
cd phase-5-6-rag-application/frontend
npm run dev

# Test build locally
npm run build

# Test production build locally
npm run start

# Check environment variables
npm run env
```

### Browser Console Testing

```javascript
// Test API connection
fetch('https://rag-backend.onrender.com/api/v1/monitoring/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Test WebSocket connection
const ws = new WebSocket('wss://rag-backend.onrender.com/ws');
ws.onopen = () => console.log('WebSocket connected');
ws.onmessage = (event) => console.log('Received:', event.data);
```

---

## Post-Deployment Checklist

### Verification
- [ ] Frontend is live and accessible
- [ ] All pages load correctly
- [ ] Dark theme is applied properly
- [ ] Navigation works smoothly
- [ ] API endpoints are accessible
- [ ] WebSocket connections work
- [ ] No console errors
- [ ] Mobile responsive design

### Performance
- [ ] Page load time < 3 seconds
- [ ] Core Web Vitals passing
- [ ] Bundle size optimized
- [ ] Images optimized
- [ ] Caching configured

### Security
- [ ] HTTPS enforced
- [ ] No sensitive data in client-side code
- [ ] CSP headers configured
- [ ] Rate limiting in place
- [ ] API keys secured

### Integration
- [ ] Backend API connectivity
- [ ] WebSocket functionality
- [ ] GraphQL endpoints working
- [ ] Error handling working
- [ ] Loading states appropriate

---

## Next Steps

### 1. GitHub Actions Integration
- Configure automatic deployments
- Set up preview deployments for PRs
- Configure deployment hooks

### 2. Monitoring Setup
- Configure error tracking
- Set up performance monitoring
- Implement user analytics

### 3. Optimization
- Implement code splitting
- Optimize bundle size
- Configure caching strategies

### 4. Documentation Update
- Update user documentation
- Document deployment process
- Create troubleshooting guide

---

## Support Resources

### Vercel Documentation
- [Vercel Docs](https://vercel.com/docs)
- [Next.js on Vercel](https://vercel.com/docs/frameworks/nextjs)
- [Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Domain Configuration](https://vercel.com/docs/concepts/projects/custom-domains)

### Common Issues
- [Troubleshooting Guide](https://vercel.com/docs/concepts/troubleshooting)
- [Performance Optimization](https://vercel.com/docs/concepts/performance)
- [Security Best Practices](https://vercel.com/docs/concepts/security)

### Community Support
- [Vercel Discord](https://vercel.com/discord)
- [GitHub Discussions](https://github.com/vercel/vercel/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/vercel)

---

## Cost Information

### Free Tier Limits
- **Bandwidth**: 100GB/month
- **Builds**: 100 per month
- **Function Invocations**: 100,000 per month
- **Serverless Function Execution**: 10,000 GB-hours/month

### Pro Tier ($20/month)
- **Bandwidth**: 1TB/month
- **Builds**: Unlimited
- **Function Invocations**: 1,000,000 per month
- **Serverless Function Execution**: 100,000 GB-hours/month
- **Analytics**: Advanced analytics
- **Team Collaboration**: Multiple team members

### Cost Optimization Tips
- Optimize bundle size
- Implement efficient caching
- Use ISR for dynamic content
- Monitor bandwidth usage
- Optimize function execution time

---

## Environment Variables Reference

### Required Variables
```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://rag-backend.onrender.com
NEXT_PUBLIC_WS_URL=wss://rag-backend.onrender.com/ws
NEXT_PUBLIC_GRAPHQL_URL=https://rag-backend.onrender.com/graphql

# Application Configuration
NEXT_PUBLIC_APP_NAME=RAG System
NEXT_PUBLIC_APP_VERSION=2.0.0
NEXT_PUBLIC_ENVIRONMENT=production
```

### Optional Variables
```bash
# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_ERROR_REPORTING=true
NEXT_PUBLIC_ENABLE_PERFORMANCE_MONITORING=true

# Configuration
NEXT_PUBLIC_API_TIMEOUT=30000
NEXT_PUBLIC_WS_RECONNECT_ATTEMPTS=5
NEXT_PUBLIC_CACHE_TTL=3600
```

---

## URLs and Endpoints

### Production URLs
- **Frontend**: `https://rag-frontend.vercel.app`
- **Backend**: `https://rag-backend.onrender.com`
- **API Docs**: `https://rag-backend.onrender.com/docs`
- **GraphQL**: `https://rag-backend.onrender.com/graphql`

### Testing Endpoints
- **Health Check**: `https://rag-backend.onrender.com/api/v1/monitoring/health`
- **WebSocket**: `wss://rag-backend.onrender.com/ws`
- **Chat API**: `https://rag-backend.onrender.com/api/v1/chat/chat`
- **Search API**: `https://rag-backend.onrender.com/api/v1/search/search`

---

## Performance Metrics

### Target Metrics
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1
- **First Input Delay**: < 100ms
- **Time to Interactive**: < 3.8s

### Monitoring Tools
- **Vercel Analytics**: Built-in performance monitoring
- **Chrome DevTools**: Local performance testing
- **Lighthouse**: Performance auditing
- **WebPageTest**: External performance testing
