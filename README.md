# 🔬 Skin Type Analyzer

A web application and API for classifying skin types (dry vs oily) using deep learning.

## 🌟 Features

- **Web Interface**: Beautiful, responsive web UI for image upload and analysis
- **REST API**: JSON API endpoint for programmatic access (perfect for Android apps)
- **Deep Learning**: Uses ResNet50 model trained on skin type classification
- **Cloud Ready**: Optimized for Google Cloud Platform deployment
- **Real-time Analysis**: Fast prediction with confidence scores

## 🚀 Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the Model**
   ```bash
   python test_model.py
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   Navigate to `http://localhost:8080`

### Google Cloud Deployment

1. **Install Google Cloud SDK**
   - Download from: https://cloud.google.com/sdk/docs/install

2. **Deploy with One Command**
   ```bash
   ./deploy.sh
   ```

3. **Access Your App**
   - The script will provide the deployed URL
   - Use the API endpoint for Android integration

## 📡 API Usage

### Endpoint
```
POST /api/predict
```

### Request
- **Content-Type**: `multipart/form-data`
- **Body**: Image file with key `image`

### Response
```json
{
  "success": true,
  "result": {
    "prediction": "dry",
    "confidence": 0.8934,
    "probabilities": {
      "dry": 0.8934,
      "oily": 0.1066
    },
    "class_index": 0
  }
}
```

### Android Integration Example

```java
// Example HTTP request for Android
OkHttpClient client = new OkHttpClient();

RequestBody requestBody = new MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("image", "image.jpg",
        RequestBody.create(MediaType.parse("image/jpeg"), imageFile))
    .build();

Request request = new Request.Builder()
    .url("https://your-project.uc.r.appspot.com/api/predict")
    .post(requestBody)
    .build();

Response response = client.newCall(request).execute();
String jsonResponse = response.body().string();
```

## 🏗️ Architecture

- **Framework**: Flask (Python)
- **Model**: ResNet50 with custom classification head
- **Classes**: 2 (dry, oily)
- **Input**: RGB images (224x224)
- **Output**: JSON with predictions and confidence scores

## 📁 Project Structure

```
kaggle_image_analyzer/
├── app.py                      # Main Flask application
├── test_model.py              # Model testing script
├── deploy.sh                  # Google Cloud deployment script
├── requirements.txt           # Python dependencies
├── app.yaml                   # Google Cloud configuration
├── .gcloudignore             # Files to exclude from deployment
├── templates/
│   └── index.html            # Web interface template
├── best_skin_model.pth       # Trained model weights
├── best_skin_model_entire.pth # Complete model (backup)
├── label_maps.pkl            # Label mappings
└── training_stats.pkl        # Training statistics
```

## 🎯 Model Details

- **Architecture**: ResNet50 pretrained on ImageNet
- **Fine-tuning**: Custom classification head for 2 classes
- **Input Size**: 224x224 RGB images
- **Normalization**: ImageNet standard normalization
- **Classes**: 
  - 0: Dry skin
  - 1: Oily skin

## 🌐 Web Interface Features

- **Drag & Drop**: Easy image upload
- **Real-time Preview**: See your image before analysis
- **Beautiful Results**: Visual confidence bars and probability breakdown
- **JSON Display**: Raw API response for developers
- **Mobile Responsive**: Works on all devices

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8080)

### Google Cloud Settings
- **Runtime**: Python 3.9
- **Memory**: 4GB
- **CPU**: 2 cores
- **Auto-scaling**: 0-10 instances

## 🚨 Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing image, invalid format)
- `500`: Server error (model prediction failed)

## 📊 Supported Image Formats

- PNG
- JPG/JPEG
- GIF
- BMP

## 🔒 Security

- File type validation
- Size limits enforced by cloud platform
- No persistent storage of uploaded images

## 🤖 Android App Integration

The API is designed to work seamlessly with Android applications:

1. **Capture Image**: Use camera or gallery
2. **Send to API**: POST multipart request to `/api/predict`
3. **Parse Response**: Handle JSON response with predictions
4. **Display Results**: Show skin type classification to user

## 📈 Performance

- **Cold Start**: ~3-5 seconds (Google Cloud)
- **Warm Requests**: ~1-2 seconds
- **Model Size**: ~100MB
- **Memory Usage**: ~2-3GB during inference

## 🛠️ Development

### Local Testing
```bash
# Test model loading
python test_model.py

# Run development server
python app.py
```

### Adding New Features
1. Modify `app.py` for new endpoints
2. Update `templates/index.html` for UI changes
3. Test locally before deployment

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Made with ❤️ for skin type classification**
