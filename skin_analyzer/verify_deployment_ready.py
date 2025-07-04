#!/usr/bin/env python3
"""
Pre-Deployment Verification Script
=================================

This script checks if everything is ready for Google Cloud deployment.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_file_exists(file_path, description):
    """Check if a required file exists."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def check_gcloud_installed():
    """Check if Google Cloud SDK is installed."""
    try:
        result = subprocess.run(['gcloud', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Google Cloud SDK is installed")
            return True
        else:
            print("❌ Google Cloud SDK not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Google Cloud SDK not installed")
        return False

def check_gcloud_auth():
    """Check if user is authenticated with Google Cloud."""
    try:
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            print("✅ Google Cloud authenticated")
            return True
        else:
            print("❌ Not authenticated with Google Cloud")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Cannot check Google Cloud authentication")
        return False

def check_model_files():
    """Check if model files exist."""
    model_files = [
        'random_forest_optimized.pkl',
        'ensemble_optimized.pkl',
        'gradient_boost_optimized.pkl',
        'scaler.pkl',
        'label_encoder.pkl'
    ]
    
    found_models = 0
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Model file: {model_file}")
            found_models += 1
        else:
            print(f"⚠️ Model file: {model_file} (optional)")
    
    if found_models > 0:
        print(f"✅ Found {found_models} model files")
        return True
    else:
        print("❌ No model files found!")
        return False

def check_deployment_files():
    """Check if all deployment files are ready."""
    files_to_check = [
        ('api_production.py', 'Production API script'),
        ('requirements_production.txt', 'Production requirements'),
        ('Dockerfile', 'Docker configuration'),
        ('deploy_to_gcloud.sh', 'Deployment script'),
        ('.gcloudignore', 'Google Cloud ignore file')
    ]
    
    all_files_exist = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    return all_files_exist

def estimate_costs():
    """Provide cost estimates for Google Cloud usage."""
    print("\n💰 Google Cloud Cost Estimates (Free Tier):")
    print("=" * 45)
    print("📊 Free Tier Limits:")
    print("   • 2 million requests/month")
    print("   • 400,000 GB-seconds compute time")
    print("   • 200,000 CPU-seconds")
    print("   • 5GB network egress (North America)")
    print("")
    print("🎯 Expected Usage (Skin Analyzer):")
    print("   • ~1000 requests/day = 30K/month (well within limit)")
    print("   • ~2 seconds per request = 60K CPU-seconds/month")
    print("   • ~100KB response size = ~3GB egress/month")
    print("")
    print("✅ Conclusion: Should stay within free tier limits!")

def provide_next_steps():
    """Provide next steps based on verification results."""
    print("\n🚀 Next Steps:")
    print("=" * 15)
    print("1. Create Google Cloud Project:")
    print("   - Go to https://console.cloud.google.com/")
    print("   - Create new project or select existing one")
    print("   - Note down your PROJECT_ID")
    print("")
    print("2. Run deployment:")
    print("   ./deploy_to_gcloud.sh")
    print("")
    print("3. Test deployed API:")
    print("   python test_deployed_api.py")
    print("")
    print("4. Integrate with mobile app:")
    print("   - Use the deployed API URL")
    print("   - Implement image upload to /analyze endpoint")

def main():
    """Main verification function."""
    print("🔍 Pre-Deployment Verification")
    print("=" * 35)
    
    all_checks_passed = True
    
    # Check deployment files
    print("\n📁 Checking deployment files...")
    if not check_deployment_files():
        all_checks_passed = False
    
    # Check model files
    print("\n🤖 Checking model files...")
    if not check_model_files():
        all_checks_passed = False
    
    # Check Google Cloud SDK
    print("\n☁️ Checking Google Cloud setup...")
    if not check_gcloud_installed():
        all_checks_passed = False
        print("   Install from: https://cloud.google.com/sdk/docs/install")
    
    if not check_gcloud_auth():
        all_checks_passed = False
        print("   Run: gcloud auth login")
    
    # Show cost estimates
    estimate_costs()
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All checks passed! Ready for deployment!")
        provide_next_steps()
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   • Install Google Cloud SDK")
        print("   • Run: gcloud auth login")
        print("   • Ensure model files exist")
    
    print("\n📚 Full guide: GOOGLE_CLOUD_DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main()
