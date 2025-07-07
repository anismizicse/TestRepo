#!/bin/bash

echo "🔧 Google Cloud Setup Check"
echo "============================"

# Check gcloud installation
if ! command -v gcloud &> /dev/null; then
    echo "❌ Google Cloud SDK is not installed"
    echo "📥 Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ Google Cloud SDK is installed"

# Check authentication
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -n 1)
if [ -z "$CURRENT_ACCOUNT" ]; then
    echo "🔑 Please authenticate with Google Cloud:"
    echo "   gcloud auth login"
    echo "   gcloud auth application-default login"
    exit 1
fi

echo "✅ Authenticated as: $CURRENT_ACCOUNT"

# Check project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "📝 Please set your Google Cloud project:"
    echo "   gcloud config set project YOUR_PROJECT_ID"
    echo ""
    echo "💡 If you don't have a project, create one at:"
    echo "   https://console.cloud.google.com/projectcreate"
    exit 1
fi

echo "✅ Project: $PROJECT_ID"

# Check App Engine
APP_EXISTS=$(gcloud app describe --project=$PROJECT_ID 2>/dev/null | grep "^id:" | cut -d' ' -f2)
if [ -z "$APP_EXISTS" ]; then
    echo "⚠️  App Engine application not created yet"
    echo "🏗️  This will be created during first deployment"
else
    echo "✅ App Engine: $APP_EXISTS"
fi

# Check required APIs
echo ""
echo "📋 Checking required APIs..."

APIS_TO_CHECK=(
    "appengine.googleapis.com"
    "cloudbuild.googleapis.com"
)

for api in "${APIS_TO_CHECK[@]}"; do
    if gcloud services list --enabled --filter="name:$api" --format="value(name)" 2>/dev/null | grep -q "$api"; then
        echo "✅ $api"
    else
        echo "❌ $api (not enabled)"
        echo "   Enable with: gcloud services enable $api"
    fi
done

echo ""
echo "🚀 Ready for deployment!"
echo "📝 Run: ./deploy.sh"
