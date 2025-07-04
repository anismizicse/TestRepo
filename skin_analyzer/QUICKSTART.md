# Skin Type Analyzer - Quick Start Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd skin_analyzer
pip install -r requirements.txt
```

### 2. Create Dataset
```bash
python utils/dataset_downloader.py
```

### 3. Train Model
```bash
python models/train_model.py
```

### 4. Make Predictions
```bash
# Single image
python predict.py --image path/to/your/image.jpg --detailed

# Multiple images
python predict.py --batch image1.jpg image2.jpg image3.jpg

# Camera input
python predict.py --camera

# Specific model architecture
python predict.py --image test.jpg --architecture efficientnet
```

### 5. Evaluate Model
```bash
python evaluate_model.py --architecture efficientnet
```

## 📊 Model Architectures

- **EfficientNet**: Best accuracy, medium speed
- **MobileNet**: Good accuracy, fastest speed
- **ResNet50**: Good accuracy, medium speed  
- **Custom CNN**: Basic architecture, fast training

## 🎯 Skin Types Detected

1. **Normal**: Balanced skin with good hydration
2. **Dry**: Lacks moisture, may appear tight or flaky
3. **Oily**: Excessive sebum production, shiny appearance
4. **Combination**: Mix of oily and dry areas
5. **Sensitive**: Reactive skin prone to irritation

## 📈 Expected Performance

- Training Accuracy: ~85-90%
- Validation Accuracy: ~80-85%
- Test Accuracy: ~78-83%

## 🛠️ Troubleshooting

### Common Issues:

1. **Memory Error**: Reduce batch size in `utils/data_loader.py`
2. **No Face Detected**: Image preprocessing will use center crop
3. **Low Accuracy**: Increase dataset size or training epochs
4. **Slow Training**: Use MobileNet architecture for faster training

### Memory Optimization:
```python
# In train_model.py, modify batch size
self.config = {
    'batch_size': 16,  # Reduce from 32
    # ... other configs
}
```

## 📁 Project Structure

```
skin_analyzer/
├── data/                   # Dataset directory
├── models/                 # Model architectures and training
├── utils/                  # Utilities for data processing
├── predict.py             # Prediction interface
├── evaluate_model.py      # Model evaluation
├── demo.py               # Demo script
└── requirements.txt      # Dependencies
```

## 🔧 API Usage

```python
from predict import SkinTypePredictor

# Initialize predictor
predictor = SkinTypePredictor()

# Analyze image
result = predictor.analyze_skin_characteristics('image.jpg')
print(f"Skin Type: {result['skin_type']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.12+
- OpenCV 4.8+
- 4GB+ RAM recommended
- GPU optional but recommended for training
