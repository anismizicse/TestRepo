#!/bin/bash
# Quick Start Script for Model Retraining
# This script provides a simple interface for retraining models

echo "🚀 Skin Classification Model Retraining - Quick Start"
echo "====================================================="

# Check if we're in the right directory
if [ ! -f "train_models_for_production.py" ]; then
    echo "❌ Error: Please run this script from the skin_analyzer directory"
    exit 1
fi

echo ""
echo "📋 Choose your workflow:"
echo "1. 🛠️  Prepare new dataset (organize images)"
echo "2. 🔄 Retrain with existing dataset"
echo "3. 🧪 Test current models"
echo "4. 📊 View training results"
echo "5. 📖 Show help/documentation"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🛠️  Starting Dataset Preparation..."
        echo "This will help you organize your new images by skin type."
        echo ""
        python3 prepare_dataset.py
        ;;
    2)
        echo ""
        echo "🔄 Starting Model Retraining..."
        echo "This will retrain your models with the latest dataset."
        echo ""
        python3 retrain_and_deploy.py
        ;;
    3)
        echo ""
        echo "🧪 Testing Current Models..."
        python3 quick_model_test.py
        ;;
    4)
        echo ""
        echo "📊 Viewing Training Results..."
        if [ -f "training_report.json" ]; then
            echo "Latest Training Report:"
            echo "====================="
            python3 -c "
import json
with open('training_report.json', 'r') as f:
    report = json.load(f)
print(f'Training Date: {report[\"timestamp\"]}')
print(f'Best Model: {report[\"best_model\"]}')
print(f'Best Accuracy: {report[\"best_accuracy\"]:.1%}')
print(f'Total Samples: {report[\"total_samples\"]:,}')
print(f'Features: {report[\"feature_count\"]}')
print(f'Augmentation: {report[\"augmentation_used\"]}')
print()
print('Model Performance:')
for model, results in report['model_results'].items():
    print(f'  {model}: {results[\"test_score\"]:.1%} accuracy')
"
        else
            echo "❌ No training report found. Please train models first."
        fi
        ;;
    5)
        echo ""
        echo "📖 Documentation and Help"
        echo "========================"
        echo ""
        echo "📂 Directory Structure:"
        echo "  new_training_data/     - Place your new images here"
        echo "  ├── combination/       - Images of combination skin"
        echo "  ├── dry/              - Images of dry skin"
        echo "  ├── normal/           - Images of normal skin"
        echo "  ├── oily/             - Images of oily skin"
        echo "  └── sensitive/        - Images of sensitive skin"
        echo ""
        echo "📋 Complete Workflow:"
        echo "  1. Organize images using option 1"
        echo "  2. Retrain models using option 2"
        echo "  3. Test models using option 3"
        echo "  4. Deploy to production"
        echo ""
        echo "📚 Detailed guides available:"
        echo "  - RETRAINING_GUIDE.md (complete step-by-step guide)"
        echo "  - TRAINING_COMPLETION_REPORT.md (current model details)"
        echo ""
        echo "🆘 Need help? Check the troubleshooting section in RETRAINING_GUIDE.md"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Task completed!"
echo "📖 For detailed information, check RETRAINING_GUIDE.md"
