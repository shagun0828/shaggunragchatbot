# Vercel Frontend Deployment Checklist

## Pre-Deployment Checklist
- [ ] Vercel account created and signed in
- [ ] GitHub repository connected to Vercel
- [ ] Backend deployed on Render and accessible
- [ ] vercel.json file in frontend directory
- [ ] package.json file exists and is complete
- [ ] Next.js frontend code ready in phase-5-6-rag-application/frontend

## Deployment Steps
- [ ] Go to https://vercel.com and sign in
- [ ] Click "New Project" > "Import Git Repository"
- [ ] Connect to shagun0828/shaggunragchatbot repository
- [ ] Set root directory to "phase-5-6-rag-application/frontend"
- [ ] Verify Next.js framework is auto-detected
- [ ] Configure environment variables
- [ ] Click "Deploy"
- [ ] Wait for deployment to complete
- [ ] Verify deployment is successful

## Environment Variables to Add
```
NEXT_PUBLIC_API_URL=https://rag-backend.onrender.com
NEXT_PUBLIC_WS_URL=wss://rag-backend.onrender.com/ws
NEXT_PUBLIC_GRAPHQL_URL=https://rag-backend.onrender.com/graphql
NEXT_PUBLIC_APP_NAME=RAG System
NEXT_PUBLIC_APP_VERSION=2.0.0
NEXT_PUBLIC_ENVIRONMENT=production
```

## Post-Deployment Verification
- [ ] Frontend URL accessible: https://rag-frontend.vercel.app
- [ ] Dark theme applied correctly
- [ ] Navigation between components works
- [ ] Chat interface loads and connects to backend
- [ ] Search interface functions properly
- [ ] Dashboard displays data correctly
- [ ] WebSocket connections established
- [ ] No console errors in browser
- [ ] Mobile responsive design works
- [ ] API endpoints responding correctly

## Testing Checklist
- [ ] Page load time < 3 seconds
- [ ] Core Web Vitals passing
- [ ] API connectivity working
- [ ] WebSocket functionality working
- [ ] GraphQL queries working
- [ ] Error handling working properly
- [ ] Loading states appropriate
- [ ] Cross-browser compatibility

## URLs to Test After Deployment
- Frontend: https://rag-frontend.vercel.app
- Backend API: https://rag-backend.onrender.com/api/v1/monitoring/health
- API Docs: https://rag-backend.onrender.com/docs
- WebSocket: wss://rag-backend.onrender.com/ws
- GraphQL: https://rag-backend.onrender.com/graphql

## Performance Metrics to Monitor
- First Contentful Paint < 1.5s
- Largest Contentful Paint < 2.5s
- Cumulative Layout Shift < 0.1
- First Input Delay < 100ms
- Time to Interactive < 3.8s

## Common Issues and Solutions
- Build failures: Check package.json and dependencies
- API connection issues: Verify environment variables
- WebSocket issues: Check WebSocket URL configuration
- CORS issues: Verify backend CORS settings
- Environment variable issues: Ensure NEXT_PUBLIC_ prefix

## Next Steps After Deployment
- [ ] Configure custom domain (optional)
- [ ] Set up analytics and monitoring
- [ ] Configure preview deployments for PRs
- [ ] Set up automatic deployments
- [ ] Test all features thoroughly
- [ ] Optimize performance if needed
