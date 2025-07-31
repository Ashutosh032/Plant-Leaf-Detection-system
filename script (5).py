# Create a complete plant leaf classification system with high accuracy
# This code will create all necessary files for a complete project

import os

# Create the main project structure
project_structure = """
Plant_Leaf_Classification/
├── main.py
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── model_architecture.py
│   └── train_model.py
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── data_augmentation.py
│   └── evaluation.py
├── config/
│   ├── __init__.py
│   └── config.py
├── data/
│   └── placeholder.txt
└── saved_models/
    └── placeholder.txt
"""

print("Project Structure:")
print(project_structure)

# Create main.py - The entry point
main_py_content = '''
"""
Plant Leaf Classification System
High-accuracy deep learning model for plant leaf identification
Author: AI Assistant
"""

import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from models.model_architecture import create_enhanced_model
from utils.data_preprocessing import load_and_preprocess_data
from utils.data_augmentation import create_data_generators
from utils.evaluation import evaluate_model, plot_training_history
from config.config import Config

def main():
    """Main function to run the plant leaf classification system"""
    
    print("🌿 Plant Leaf Classification System")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Check GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    try:
        # Step 1: Load and preprocess data
        print("\\n📁 Loading and preprocessing data...")
        train_generator, validation_generator, test_generator, num_classes = load_and_preprocess_data()
        
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Test samples: {test_generator.samples}")
        
        # Step 2: Create the model
        print("\\n🏗️ Creating enhanced model...")
        model = create_enhanced_model(num_classes)
        
        # Step 3: Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=Config.INITIAL_LEARNING_RATE,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Print model summary
        print("\\n📊 Model Summary:")
        model.summary()
        
        # Step 4: Set up callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath='saved_models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Learning rate scheduler
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: Config.INITIAL_LEARNING_RATE * (0.95 ** epoch)
            )
        ]
        
        # Step 5: Train the model
        print("\\n🚀 Starting training...")
        history = model.fit(
            train_generator,
            epochs=Config.EPOCHS,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Step 6: Load best model and evaluate
        print("\\n📈 Loading best model and evaluating...")
        best_model = tf.keras.models.load_model('saved_models/best_model.h5')
        
        # Evaluate on test set
        test_results = evaluate_model(best_model, test_generator)
        
        # Step 7: Plot training history
        plot_training_history(history)
        
        # Step 8: Save final model
        best_model.save('saved_models/final_plant_classifier.h5')
        print("\\n✅ Model saved successfully!")
        
        # Step 9: Print final results
        print("\\n🎯 Final Results:")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test Top-3 Accuracy: {test_results.get('top_3_accuracy', 'N/A')}")
        
        return best_model
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

def predict_single_image(model_path, image_path, class_names):
    """Predict the class of a single leaf image"""
    
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    img_array /= 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [(class_names[i], predictions[0][i]) for i in top_3_indices]
    
    print(f"\\n🔍 Prediction Results:")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\\nTop 3 predictions:")
    for i, (class_name, prob) in enumerate(top_3_predictions, 1):
        print(f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
    
    return predicted_class, confidence, top_3_predictions

if __name__ == "__main__":
    # Train the model
    trained_model = main()
    
    # Example usage for prediction
    if trained_model is not None:
        print("\\n🎯 System ready for predictions!")
        print("Use predict_single_image() function to classify new images.")
'''

# Save main.py content
with open('main_py_content.txt', 'w') as f:
    f.write(main_py_content)

print("✅ Main.py content created!")