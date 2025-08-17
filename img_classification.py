import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

class CatDogClassifier:
    def __init__(self, img_height=150, img_width=150):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create CNN model architecture"""
        self.model = models.Sequential([
            # First C
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D(2, 2),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Fourth Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model Architecture:")
        self.model.summary()
        
    def prepare_data(self, train_dir, validation_dir, batch_size=32):
        """Prepare training and validation data with augmentation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='binary'
        )
        
        return train_generator, validation_generator
    
    def train_model(self, train_generator, validation_generator, epochs=25):
        """Train the CNN model"""
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks
        )
        
        return self.history
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict_image(self, img_path):
        """Predict single image"""
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array /= 255.0  # Normalize
        
        prediction = self.model.predict(img_array)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            return f"🐕 DOG (Confidence: {confidence:.2%})"
        else:
            return f"🐱 CAT (Confidence: {1-confidence:.2%})"
    
    def save_model(self, filepath):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def create_sample_dataset_structure():
    """Create sample dataset directory structure"""
    directories = [
        'dataset/train/cats',
        'dataset/train/dogs', 
        'dataset/validation/cats',
        'dataset/validation/dogs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Dataset directory structure created!")
    print("Please add your images to the following folders:")
    for directory in directories:
        print(f"  - {directory}")


if __name__ == "__main__":
    print("🐱🐶 Cats vs Dogs Image Classification")
    print("=" * 50)
    
    # Initialize classifier
    classifier = CatDogClassifier()
    
    # Create model
    classifier.create_model()
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("\n📁 Creating dataset structure...")
        create_sample_dataset_structure()
        print("\nPlease add your images and run again!")
    else:
        # Prepare data
        print("\n📊 Preparing data...")
        train_gen, val_gen = classifier.prepare_data(
            'dataset/train', 
            'dataset/validation'
        )
        
        # Train model
        print("\n🚀 Training model...")
        history = classifier.train_model(train_gen, val_gen, epochs=25)
        
        # Plot results
        print("\n📈 Plotting training history...")
        classifier.plot_training_history()
        
        # Save model
        classifier.save_model('cats_dogs_model.h5')
        
        # Example prediction (if test image exists)
        test_image = 'test_image.jpg'
        if os.path.exists(test_image):
            result = classifier.predict_image(test_image)
            print(f"\n🔍 Prediction for {test_image}: {result}")
        
        print("\n✅ Training complete!")
        print("Use classifier.predict_image('path_to_image.jpg') to classify new images!")