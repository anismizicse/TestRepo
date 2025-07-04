# Google Cloud Deployment Guide

## 📋 Quick Start

For immediate deployment, follow these essential steps:

### Prerequisites
1. Install Google Cloud CLI
2. Create/select a Google Cloud project
3. Enable Cloud Run API and Container Registry API

### Simple Deployment
```bash
# Login and configure
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy using our script
chmod +x deployment/deploy_to_gcloud.sh
./deployment/deploy_to_gcloud.sh
```

---

## 🔧 Complete Setup Guide

### 1. Install Google Cloud SDK

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install google-cloud-cli
```

#### macOS
```bash
brew install google-cloud-sdk
```

#### Windows
Download installer from [Google Cloud website](https://cloud.google.com/sdk/docs/install)

### 2. Authenticate and Setup Project
```bash
# Login to Google Cloud
gcloud auth login

# Create or select project
gcloud projects create YOUR_PROJECT_ID  # if creating new
gcloud config set project YOUR_PROJECT_ID

# Enable billing (required for deployment)
# Visit: https://console.cloud.google.com/billing
```

### 3. Enable Required APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 🚀 Deployment Methods

### Method 1: Automated Script (Recommended)
```bash
# Update project ID in deployment script
nano deployment/deploy_to_gcloud.sh
# Change: PROJECT_ID="your-project-id"

# Make executable and run
chmod +x deployment/deploy_to_gcloud.sh
./deployment/deploy_to_gcloud.sh
```

### Method 2: Manual Deployment
```bash
gcloud run deploy skin-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --min-instances 0
```

### Method 3: Using Docker
```bash
# Build image
docker build -t gcr.io/YOUR_PROJECT_ID/skin-analyzer .

# Push to registry
docker push gcr.io/YOUR_PROJECT_ID/skin-analyzer

# Deploy
gcloud run deploy skin-analyzer \
  --image gcr.io/YOUR_PROJECT_ID/skin-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🧪 Testing Deployment

### 1. Get Service URL
```bash
gcloud run services describe skin-analyzer \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

### 2. Test Health Endpoint
```bash
curl https://YOUR_SERVICE_URL/
```

### 3. Test Image Analysis
```bash
# Upload file
curl -X POST \
  -F 'image=@test_sample_image.jpg' \
  https://YOUR_SERVICE_URL/analyze

# Base64 analysis
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image":"base64_image_data"}' \
  https://YOUR_SERVICE_URL/analyze-base64
```

## 📊 Monitoring & Maintenance

### View Logs
```bash
gcloud logs tail --service skin-analyzer
```

### Monitor Performance
```bash
# View service details
gcloud run services describe skin-analyzer --region=us-central1

# Monitor metrics in Cloud Console
# Visit: https://console.cloud.google.com/run
```

### Common Monitoring Points
- **Response Times**: Keep under 2-3 seconds
- **Error Rates**: Monitor 4xx/5xx responses
- **Memory Usage**: Watch for memory spikes
- **CPU Usage**: Monitor processing load

## ⚡ Performance Optimization

### Reduce Cold Starts
```yaml
# In app.yaml
min_instances: 1  # Keep at least 1 instance warm
```

### Improve Concurrency
```yaml
# Handle more requests per instance
concurrency: 80   # Requests per instance
```

### Memory Management
- Monitor memory usage in Cloud Console
- Optimize image processing pipeline
- Consider model compression if needed

## 🔧 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Dockerfile syntax
   - Verify all dependencies in requirements.txt
   - Ensure model files are included

2. **Memory Issues**
   - Increase memory allocation in deployment
   - Check for memory leaks in image processing

3. **Model Loading Errors**
   - Verify model files exist in deployment
   - Check file permissions and paths

### Debugging Commands
```bash
# View detailed logs
gcloud logs tail --service skin-analyzer --format="value(textPayload)"

# Check deployment status
gcloud run services describe skin-analyzer --region=us-central1

# Test locally before deployment
python app/main.py
```

## 💰 Cost Optimization

### Resource Management
- Use `min_instances: 0` for development/testing
- Monitor usage patterns and adjust resources
- Consider regional deployment for better latency

### Cost Monitoring
- Set up billing alerts in Google Cloud Console
- Review monthly usage reports
- Optimize based on actual traffic patterns

## 🔄 Update Process

### For Code Changes
1. Test changes locally
2. Deploy to staging environment (optional)
3. Run integration tests
4. Deploy to production
5. Monitor for issues

### For Model Updates
1. Follow the retraining guide (see RETRAINING_GUIDE.md)
2. Test new models locally
3. Deploy with new model files
4. Compare performance metrics

## 📚 Support Resources

- **Google Cloud Run Documentation**: https://cloud.google.com/run/docs
- **Cloud Console**: https://console.cloud.google.com/run
- **Cost Calculator**: https://cloud.google.com/products/calculator
- **Support Forum**: https://stackoverflow.com/questions/tagged/google-cloud-run

## 📋 Maintenance Schedule

### Regular Tasks
- **Weekly**: Check logs for errors
- **Monthly**: Review usage and costs
- **Quarterly**: Update dependencies
- **As needed**: Retrain models with new data
- **As needed**: Scale resources based on usage

---

*For model retraining instructions, see RETRAINING_GUIDE.md*
*For training performance details, see TRAINING_COMPLETION_REPORT.md*
