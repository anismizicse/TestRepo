"""
Skin Type Classification Model Architecture
Deep learning model for analyzing skin types from facial images
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np


class SkinClassifier:
    """
    CNN-based skin type classifier with transfer learning capabilities
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=5, 
                 base_model='efficientnet', learning_rate=0.001):
        """
        Initialize the skin classifier
        
        Args:
            input_shape (tuple): Input image shape
            num_classes (int): Number of skin type classes
            base_model (str): Base model architecture ('efficientnet', 'resnet50', 'mobilenet', 'custom')
            learning_rate (float): Learning rate for training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_model_name = base_model
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # Skin type class names
        self.class_names = ['normal', 'dry', 'oily', 'combination', 'sensitive']
    
    def build_model(self):
        """
        Build the skin classification model
        
        Returns:
            tf.keras.Model: Compiled model
        """
        if self.base_model_name == 'efficientnet':
            self.model = self._build_efficientnet_model()
        elif self.base_model_name == 'resnet50':
            self.model = self._build_resnet_model()
        elif self.base_model_name == 'mobilenet':
            self.model = self._build_mobilenet_model()
        elif self.base_model_name == 'custom':
            self.model = self._build_custom_cnn_model()
        else:
            raise ValueError(f"Unknown base model: {self.base_model_name}")
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        return self.model
    
    def _build_efficientnet_model(self):
        """
        Build model based on EfficientNetB0
        """
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def _build_resnet_model(self):
        """
        Build model based on ResNet50
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def _build_mobilenet_model(self):
        """
        Build model based on MobileNetV2 (lightweight)
        """
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def _build_custom_cnn_model(self):
        """
        Build custom CNN model from scratch
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global average pooling and classification head
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def get_callbacks(self, model_save_path, patience=10):
        """
        Get training callbacks
        
        Args:
            model_save_path (str): Path to save the best model
            patience (int): Early stopping patience
            
        Returns:
            list: List of callbacks
        """
        callbacks = [
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def fine_tune_model(self, train_dataset, val_dataset, epochs=10, 
                       fine_tune_at=100):
        """
        Fine-tune the pre-trained model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of fine-tuning epochs
            fine_tune_at (int): Layer from which to start fine-tuning
        """
        if self.base_model_name == 'custom':
            print("Fine-tuning not applicable for custom model")
            return
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Fine-tune from this layer onwards
        for layer in self.model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        print(f"Fine-tuning {self.model.layers[0].name} from layer {fine_tune_at}")
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=self.get_callbacks('models/saved_models/fine_tuned_model.h5'),
            verbose=1
        )
        
        # Combine histories if previous training exists
        if self.history is not None:
            for key in fine_tune_history.history.keys():
                if key in self.history.history:
                    self.history.history[key].extend(fine_tune_history.history[key])
                else:
                    self.history.history[key] = fine_tune_history.history[key]
        else:
            self.history = fine_tune_history
    
    def train(self, train_dataset, val_dataset, epochs=50, 
              model_save_path='models/saved_models/skin_classifier.h5'):
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the trained model
            
        Returns:
            tf.keras.callbacks.History: Training history
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training {self.base_model_name} model for {epochs} epochs...")
        print(f"Model summary:")
        self.model.summary()
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=self.get_callbacks(model_save_path),
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        print("Evaluating model on test dataset...")
        
        # Evaluate the model
        test_loss, test_accuracy, test_top2_accuracy = self.model.evaluate(
            test_dataset, 
            verbose=1
        )
        
        # Get predictions for detailed analysis
        predictions = self.model.predict(test_dataset)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top2_accuracy': test_top2_accuracy,
            'predictions': predictions
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return results
    
    def predict(self, image_batch):
        """
        Make predictions on a batch of images
        
        Args:
            image_batch (numpy.ndarray): Batch of preprocessed images
            
        Returns:
            tuple: (predicted_classes, confidence_scores, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Get predictions
        probabilities = self.model.predict(image_batch)
        
        # Get predicted classes and confidence scores
        predicted_classes = np.argmax(probabilities, axis=1)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Convert class indices to class names
        predicted_names = [self.class_names[idx] for idx in predicted_classes]
        
        return predicted_names, confidence_scores, probabilities
    
    def predict_single_image(self, image):
        """
        Predict skin type for a single image
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            dict: Prediction results
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        predicted_names, confidence_scores, probabilities = self.predict(image_batch)
        
        result = {
            'skin_type': predicted_names[0],
            'confidence': float(confidence_scores[0]),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities[0])
            }
        }
        
        return result
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not built"
        
        # Capture summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
