import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import pickle
import os

def load_model():
    """
    Load the trained PyTorch model and class labels.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    
    # Load label mapping
    label_maps_path = os.path.join(model_dir, "label_maps.pkl")
    with open(label_maps_path, "rb") as f:
        label_maps = pickle.load(f)

    # Extract class names using the correct structure from training
    # The label_maps.pkl contains: {'label_index': {'dry': 0, 'normal': 1, 'oily': 2}, 'index_label': {0: 'dry', 1: 'normal', 2: 'oily'}}
    if 'index_label' in label_maps:
        # Use index_label mapping to get class names in correct order
        index_label = label_maps['index_label']
        class_names = [index_label[i] for i in sorted(index_label.keys())]
    else:
        # Fallback for different format
        if isinstance(next(iter(label_maps.values())), str):
            # Example: {0: "dry", 1: "normal", 2: "oily"}
            class_names = [label_maps[i] for i in sorted(label_maps.keys())]
        elif isinstance(next(iter(label_maps.values())), dict):
            # Example: {"dry": 0, "normal": 1, "oily": 2}
            class_names = list(next(iter(label_maps.values())).keys())
        else:
            raise ValueError("Unexpected format in label_maps.pkl")

    # Load the trained PyTorch model
    model_path = os.path.join(model_dir, "best_skin_model_entire.pth")
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    print(f"✅ Model loaded with {len(class_names)} classes: {class_names}")
    print(f"✅ Class order: {class_names}")
    return model, class_names

# Image preprocessing pipeline (matches training setup)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(model, image: Image.Image, class_names):
    """
    Perform prediction on a single image and return top class and probabilities.
    """
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        # Fix: Apply softmax to the entire batch output, not individual element
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # Get first (and only) batch item

    # Ensure number of classes matches model output size
    num_classes = probs.shape[0]
    
    # Ensure class_names are in the correct order (should already be fixed in load_model)
    if len(class_names) != num_classes:
        print(f"Warning: class_names length ({len(class_names)}) != model output classes ({num_classes})")
        class_names = class_names[:num_classes]

    predictions = [
        {"class": class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in range(num_classes)
    ]
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "predictions": predictions,
        "top_class": predictions[0]["class"],
        "raw_scores": [round(float(probs[i]), 6) for i in range(num_classes)]  # For debugging
    }
