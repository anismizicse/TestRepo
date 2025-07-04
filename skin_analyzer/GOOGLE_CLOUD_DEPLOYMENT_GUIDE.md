# 🚀 Complete Guide: Deploy Skin Analyzer to Google Cloud (Free Tier)

## Overview
This guide will help you deploy your skin analyzer model to Google Cloud Platform's free tier using Google Cloud Run. Your API will be accessible worldwide and can handle mobile app requests.

## 📋 Prerequisites

### 1. Google Cloud Account Setup
- Create a Google Cloud account at https://cloud.google.com/
- Get $300 free credits (valid for 90 days)
- Create a new project or use an existing one

### 2. Install Google Cloud SDK
**For macOS:**
```bash
# Install using Homebrew
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

**For Windows:**
- Download installer from https://cloud.google.com/sdk/docs/install
- Run the installer and follow instructions

**For Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

## 🔧 Step-by-Step Deployment

### Step 1: Authenticate with Google Cloud
```bash
# Login to your Google account
gcloud auth login

# Set your project ID (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# Verify authentication
gcloud auth list
```

### Step 2: Prepare Your Model Files
Make sure these files are in your skin_analyzer directory:
- ✅ `api_production.py` (created)
- ✅ `requirements_production.txt` (created)
- ✅ `Dockerfile` (created)
- ✅ `deploy_to_gcloud.sh` (created)
- ✅ `*.pkl` model files (your trained models)

### Step 3: Quick Deployment (Automated)
```bash
cd /Users/bjit/Desktop/My_Files/Projects/Image_Analyzer/skin_analyzer

# Make script executable (already done)
chmod +x deploy_to_gcloud.sh

# Run automated deployment
./deploy_to_gcloud.sh
```

### Step 4: Manual Deployment (Alternative)
If you prefer manual control:

```bash
# 1. Set your project ID
export PROJECT_ID="your-actual-project-id"
export SERVICE_NAME="skin-analyzer-api"
export REGION="us-central1"

# 2. Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# 3. Build and deploy
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME .

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 10 \
    --min-instances 0
```

## 🌐 Free Tier Limits & Optimization

### Google Cloud Run Free Tier Includes:
- **2 million requests per month**
- **400,000 GB-seconds of compute time**
- **200,000 CPU-seconds**
- **5GB network egress from North America per month**

### Our Configuration:
- **Memory**: 1GB (sufficient for ML models)
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Min instances**: 0 (cost-effective)
- **Max instances**: 10 (scalable)

## 📱 API Endpoints

Once deployed, your API will have these endpoints:

### Health Check
```bash
GET https://your-service-url.run.app/
```

### Image Analysis (File Upload)
```bash
POST https://your-service-url.run.app/analyze
Content-Type: multipart/form-data

# Form data:
image: [image file]
```

### Image Analysis (Base64)
```bash
POST https://your-service-url.run.app/analyze-base64
Content-Type: application/json

{
  "image": "base64-encoded-image-data"
}
```

## 🧪 Testing Your Deployed API

### Test with cURL:
```bash
# Health check
curl https://your-service-url.run.app/

# Upload image
curl -X POST \
  -F "image=@path/to/your/image.jpg" \
  https://your-service-url.run.app/analyze

# Base64 test (if you have base64 image data)
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image":"base64-image-data-here"}' \
  https://your-service-url.run.app/analyze-base64
```

### Expected Response:
```json
{
  "predicted_skin_type": "normal",
  "confidence": 0.85,
  "confidence_scores": {
    "combination": 0.12,
    "dry": 0.08,
    "normal": 0.85,
    "oily": 0.03,
    "sensitive": 0.02
  },
  "status": "success"
}
```

## 📊 Monitoring & Maintenance

### View Logs:
```bash
gcloud logs read --service=skin-analyzer-api --limit=50
```

### Update Deployment:
```bash
# After making changes, redeploy:
gcloud builds submit --tag gcr.io/$PROJECT_ID/skin-analyzer-api .
gcloud run deploy skin-analyzer-api --image gcr.io/$PROJECT_ID/skin-analyzer-api
```

### Check Service Status:
```bash
gcloud run services list
gcloud run services describe skin-analyzer-api --region=us-central1
```

## 🔒 Security Considerations

### For Production Use:
1. **Authentication**: Add API key authentication
2. **Rate Limiting**: Implement request rate limiting
3. **CORS**: Configure proper CORS policies
4. **Input Validation**: Enhanced image validation
5. **Monitoring**: Set up alerting and monitoring

## 💰 Cost Management

### Stay Within Free Tier:
- Monitor usage in Google Cloud Console
- Set up billing alerts
- Use `--min-instances 0` to scale to zero
- Optimize image processing for faster responses

### Cost Optimization Tips:
1. **Efficient Models**: Use compressed model files
2. **Quick Responses**: Optimize inference time
3. **Smart Scaling**: Configure appropriate scaling settings
4. **Resource Limits**: Set memory/CPU limits appropriately

## 🚨 Troubleshooting

### Common Issues:

**1. Authentication Error:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**2. API Not Enabled:**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

**3. Model Files Missing:**
- Ensure all `.pkl` files are in the directory
- Check file paths in `api_production.py`

**4. Memory Issues:**
- Increase memory limit in deployment
- Optimize model size

**5. Timeout Issues:**
- Increase timeout in Cloud Run settings
- Optimize image processing speed

## 📞 Support & Next Steps

### After Successful Deployment:
1. **Test thoroughly** with various image types
2. **Monitor performance** and costs
3. **Integrate with mobile app** using the API URL
4. **Set up monitoring** and alerts
5. **Plan for scaling** if usage grows

### For Mobile App Integration:
- Use the deployed API URL in your Flutter/Android app
- Implement proper error handling
- Add loading states for API calls
- Consider image compression before upload

---

## 🎯 Quick Start Commands

```bash
# 1. Navigate to project directory
cd /Users/bjit/Desktop/My_Files/Projects/Image_Analyzer/skin_analyzer

# 2. Login to Google Cloud
gcloud auth login

# 3. Set your project
gcloud config set project YOUR_PROJECT_ID

# 4. Deploy with one command
./deploy_to_gcloud.sh
```

That's it! Your skin analyzer API will be live and ready for mobile app integration! 🚀
