#!/bin/bash

# Deployment script for Google Cloud App Engine
echo "🚀 Deploying Skin Type Analyzer to Google Cloud Platform"
echo "========================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if we're logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 | grep -q "@"; then
    echo "🔑 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Check if project is set
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "📝 Please set your Google Cloud project ID:"
    read -p "Enter your project ID: " PROJECT_ID
    gcloud config set project $PROJECT_ID
fi

echo "📦 Project: $PROJECT_ID"

# Confirm deployment
echo ""
read -p "🤔 Deploy to Google Cloud App Engine? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "❌ Deployment cancelled."
    exit 0
fi

# Deploy the application
echo "🚀 Deploying application..."
gcloud app deploy app.yaml --quiet

# Get the deployed URL
echo ""
echo "✅ Deployment completed!"
echo "🌐 Your application is available at:"
gcloud app browse --no-launch-browser

echo ""
echo "📡 API Endpoint for Android app:"
echo "   POST https://$PROJECT_ID.uc.r.appspot.com/api/predict"
echo ""
echo "📋 Usage:"
echo "   - Web Interface: Visit the URL above"
echo "   - API: Send POST request with 'image' file to /api/predict"
echo ""
echo "🎉 Happy analyzing! 🔬"
