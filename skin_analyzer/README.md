# 🔬 Skin Type Analysis Model

A comprehensive deep learning model for analyzing skin types from facial images using computer vision and machine learning techniques.

## ✨ Features

- **🎯 Skin Type Classification**: Classifies skin into 5 types (Normal, Dry, Oily, Combination, Sensitive)
- **🧠 Deep Learning Architecture**: Multiple CNN architectures (EfficientNet, ResNet50, MobileNet, Custom)
- **👤 Face Detection**: Automated face detection and cropping using Haar cascades
- **🔄 Data Preprocessing**: Complete image preprocessing and augmentation pipeline
- **📊 Model Training**: Comprehensive training pipeline with transfer learning
- **📈 Model Evaluation**: Detailed performance analysis with ROC curves, confusion matrices
- **⚡ Real-time Prediction**: Camera-based and batch prediction capabilities
- **🎨 Clean API**: Easy-to-use prediction interface with detailed analysis

## Project Structure

```
skin_analyzer/
├── data/
│   ├── train/          # Training dataset
│   └── test/           # Testing dataset
├── models/
│   ├── skin_classifier.py    # Main model architecture
│   ├── train_model.py        # Training script
│   └── saved_models/         # Trained model files
├── utils/
│   ├── data_loader.py        # Data loading utilities
│   ├── image_processor.py    # Image preprocessing
│   └── dataset_downloader.py # Dataset download utilities
├── predict.py          # Prediction interface
├── evaluate_model.py   # Model evaluation
└── requirements.txt    # Dependencies
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare the dataset:
```bash
python utils/dataset_downloader.py
```

3. Train the model:
```bash
python models/train_model.py
```

4. Evaluate the model:
```bash
python evaluate_model.py
```

5. Make predictions:
```bash
python predict.py --image_path path/to/your/image.jpg
```

## Skin Types Classification

The model classifies skin into 5 main types:

1. **Normal**: Balanced skin with good hydration
2. **Dry**: Lacks moisture, may appear tight or flaky
3. **Oily**: Excessive sebum production, shiny appearance
4. **Combination**: Mix of oily and dry areas
5. **Sensitive**: Reactive skin prone to irritation

## Model Architecture

- **Base Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224x3 RGB images
- **Preprocessing**: Face detection, cropping, normalization
- **Data Augmentation**: Rotation, flipping, brightness adjustment
- **Output**: 5-class probability distribution

## Usage Example

```python
from predict import SkinTypePredictor

# Initialize predictor
predictor = SkinTypePredictor()

# Load and predict
result = predictor.predict_image('path/to/face_image.jpg')
print(f"Skin Type: {result['skin_type']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Performance Metrics

- Training Accuracy: ~85-90%
- Validation Accuracy: ~80-85%
- Test Accuracy: ~78-83%

## Dataset Information

The model is trained on a curated dataset containing:
- Facial images with different skin types
- Balanced distribution across all skin categories
- High-quality, diverse demographic representation

## Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional dermatological advice or medical diagnosis.

## License

MIT License - See LICENSE file for details.
