#!/bin/bash

# Google Cloud Deployment Script for Skin Analyzer API
# This script automates the deployment process to Google Cloud Run

set -e  # Exit on any error

echo "🚀 Google Cloud Skin Analyzer Deployment Script"
echo "================================================"

# Configuration
PROJECT_ID="your-project-id"  # Replace with your actual project ID
SERVICE_NAME="skin-analyzer-api"
REGION="us-central1"  # Free tier region
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed!"
        echo "Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    print_success "Google Cloud SDK found"
}

# Check if user is authenticated
check_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "Not authenticated with Google Cloud"
        print_status "Please run: gcloud auth login"
        exit 1
    fi
    print_success "Google Cloud authentication verified"
}

# Set up project
setup_project() {
    print_status "Setting up Google Cloud project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_status "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
    
    print_success "Project setup complete"
}

# Build Docker image
build_image() {
    print_status "Building Docker image..."
    
    # Build and tag the image
    gcloud builds submit --tag $IMAGE_NAME .
    
    print_success "Docker image built successfully"
}

# Deploy to Cloud Run
deploy_service() {
    print_status "Deploying to Google Cloud Run..."
    
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_NAME \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 1Gi \
        --cpu 1 \
        --max-instances 10 \
        --min-instances 0 \
        --timeout 300 \
        --concurrency 80
    
    print_success "Service deployed successfully"
}

# Get service URL
get_service_url() {
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
    print_success "Service URL: $SERVICE_URL"
    echo ""
    echo "🎉 Deployment Complete!"
    echo "=============================="
    echo "Service Name: $SERVICE_NAME"
    echo "Region: $REGION"
    echo "URL: $SERVICE_URL"
    echo ""
    echo "Test your API:"
    echo "curl $SERVICE_URL/"
    echo ""
    echo "Or upload an image:"
    echo "curl -X POST -F 'image=@your_image.jpg' $SERVICE_URL/analyze"
}

# Main execution
main() {
    echo "Starting deployment process..."
    echo ""
    
    # Prompt for project ID if not set
    if [ "$PROJECT_ID" = "your-project-id" ]; then
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        if [ -z "$PROJECT_ID" ]; then
            print_error "Project ID is required!"
            exit 1
        fi
    fi
    
    print_status "Using Project ID: $PROJECT_ID"
    echo ""
    
    # Run deployment steps
    check_gcloud
    check_auth
    setup_project
    build_image
    deploy_service
    get_service_url
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "build")
        check_gcloud
        check_auth
        build_image
        ;;
    "setup")
        check_gcloud
        check_auth
        setup_project
        ;;
    "help")
        echo "Usage: $0 [deploy|build|setup|help]"
        echo ""
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  build   - Build Docker image only"
        echo "  setup   - Setup project and APIs only"
        echo "  help    - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
