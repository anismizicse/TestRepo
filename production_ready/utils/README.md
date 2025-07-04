# Utility Scripts

This directory contains additional tools for advanced users who want to:
- Retrain the models with new data
- Evaluate model performance  
- Prepare custom datasets

## Scripts Included

### 🔄 **retrain_and_deploy.py**
- Automated script to retrain models and deploy to Google Cloud
- Useful when you have new training data
- Handles the complete pipeline from training to deployment

### 📊 **enhanced_production_trainer.py** 
- Advanced model training with ensemble methods
- Includes data augmentation and feature selection
- Use this for the best model performance

### 📁 **prepare_dataset.py**
- Organizes images into training/validation splits
- Handles data preprocessing and validation
- Run this before training new models

### 🧪 **model_evaluation_suite.py**
- Comprehensive model performance analysis
- Generates confusion matrices and metrics
- Use this to evaluate model quality

## Usage

These scripts are for advanced users only. The main application in the `app/` directory is ready to use without these tools.

For basic deployment, use the files in the root directory and `deployment/` folder.
