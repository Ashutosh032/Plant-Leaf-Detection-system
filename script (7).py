# Create data preprocessing utilities
data_preprocessing_content = '''
"""
Data preprocessing utilities for plant leaf classification
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config.config import Config

def download_plantvillage_dataset():
    """Download PlantVillage dataset if not exists"""
    import subprocess
    import zipfile
    import requests
    
    data_dir = Config.DATA_PATH
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if dataset already exists
    if os.path.exists(os.path.join(data_dir, 'PlantVillage')):
        print("Dataset already exists!")
        return os.path.join(data_dir, 'PlantVillage')
    
    try:
        # Download from Kaggle (requires Kaggle API)
        print("Downloading PlantVillage dataset from Kaggle...")
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 
            'emmarex/plantdisease', '-p', data_dir
        ], check=True)
        
        # Extract the dataset
        with zipfile.ZipFile(os.path.join(data_dir, 'plantdisease.zip'), 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file
        os.remove(os.path.join(data_dir, 'plantdisease.zip'))
        
        print("Dataset downloaded successfully!")
        return os.path.join(data_dir, 'PlantVillage')
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please manually download the PlantVillage dataset and place it in the data/ folder")
        return None

def create_data_generators():
    """Create data generators with augmentation"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=Config.ROTATION_RANGE,
        width_shift_range=Config.WIDTH_SHIFT_RANGE,
        height_shift_range=Config.HEIGHT_SHIFT_RANGE,
        shear_range=Config.SHEAR_RANGE,
        zoom_range=Config.ZOOM_RANGE,
        horizontal_flip=Config.HORIZONTAL_FLIP,
        vertical_flip=Config.VERTICAL_FLIP,
        brightness_range=Config.BRIGHTNESS_RANGE,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation and test
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    return train_datagen, test_datagen

def load_and_preprocess_data():
    """Load and preprocess the PlantVillage dataset"""
    
    # Download dataset if not exists
    dataset_path = download_plantvillage_dataset()
    if dataset_path is None:
        raise ValueError("Dataset not found. Please download manually.")
    
    # Create data generators
    train_datagen, test_datagen = create_data_generators()
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=Config.RANDOM_SEED
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=Config.RANDOM_SEED
    )
    
    # For testing, we'll use a separate test set
    # Create test generator (you might need to create a separate test folder)
    test_generator = test_datagen.flow_from_directory(
        dataset_path,
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    
    return train_generator, validation_generator, test_generator, num_classes

def visualize_dataset_samples(generator, num_samples=9):
    """Visualize sample images from the dataset"""
    
    # Get a batch of images and labels
    images, labels = next(generator)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    for i in range(min(num_samples, len(images))):
        row, col = i // 3, i % 3
        axes[row, col].imshow(images[i])
        
        # Get class name
        class_idx = np.argmax(labels[i])
        class_name = list(generator.class_indices.keys())[class_idx]
        axes[row, col].set_title(class_name.replace('_', ' '), fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset_distribution(generator):
    """Analyze and visualize class distribution"""
    
    class_counts = {}
    class_names = list(generator.class_indices.keys())
    
    # Count samples in each class
    for class_name in class_names:
        class_counts[class_name] = len(os.listdir(
            os.path.join(generator.directory, class_name)
        ))
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    names = list(class_counts.keys())
    values = list(class_counts.values())
    
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_samples = sum(values)
    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(names)}")
    print(f"Average samples per class: {total_samples / len(names):.2f}")
    print(f"Min samples: {min(values)}")
    print(f"Max samples: {max(values)}")
    
    return class_counts

def create_balanced_dataset(source_dir, target_dir, min_samples_per_class=500):
    """Create a balanced dataset by augmenting underrepresented classes"""
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
    
    # Augmentation generator for balancing
    aug_gen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    
    for class_name in os.listdir(source_dir):
        class_source = os.path.join(source_dir, class_name)
        class_target = os.path.join(target_dir, class_name)
        
        if not os.path.isdir(class_source):
            continue
            
        os.makedirs(class_target, exist_ok=True)
        
        # Copy original images
        original_images = os.listdir(class_source)
        for img_name in original_images:
            import shutil
            shutil.copy2(
                os.path.join(class_source, img_name),
                os.path.join(class_target, img_name)
            )
        
        # Augment if needed
        current_count = len(original_images)
        if current_count < min_samples_per_class:
            needed = min_samples_per_class - current_count
            print(f"Augmenting {class_name}: adding {needed} samples")
            
            # Generate augmented images
            for i, img_name in enumerate(original_images):
                if needed <= 0:
                    break
                    
                img_path = os.path.join(class_source, img_name)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Generate multiple augmented versions
                aug_iter = aug_gen.flow(img_array, batch_size=1)
                for j in range(min(5, needed)):  # Up to 5 augmented versions per image
                    aug_img = next(aug_iter)[0].astype(np.uint8)
                    aug_img_pil = tf.keras.preprocessing.image.array_to_img(aug_img)
                    
                    aug_name = f"aug_{i}_{j}_{img_name}"
                    aug_img_pil.save(os.path.join(class_target, aug_name))
                    needed -= 1
                    
                    if needed <= 0:
                        break
    
    print("Balanced dataset created successfully!")
'''

# Create evaluation utilities
evaluation_content = '''
"""
Model evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf

def evaluate_model(model, test_generator):
    """Comprehensive model evaluation"""
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions
    y_pred = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Generate classification report
    report = classification_report(
        y_true, y_pred_classes, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Print detailed results
    print("🎯 Detailed Evaluation Results:")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Test Samples: {len(y_true)}")
    
    print("\\nPer-class Results:")
    print("-" * 50)
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  Support: {metrics['support']}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    # Calculate additional metrics
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred,
        'true_labels': y_true,
        'predicted_classes': y_pred_classes
    }
    
    return results

def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Plot top-3 accuracy if available
    if 'top_3_accuracy' in history.history:
        axes[1, 1].plot(history.history['top_3_accuracy'], label='Training Top-3')
        axes[1, 1].plot(history.history['val_top_3_accuracy'], label='Validation Top-3')
        axes[1, 1].set_title('Top-3 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-3 Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_misclassifications(model, test_generator, num_examples=12):
    """Analyze and visualize misclassified examples"""
    
    test_generator.reset()
    
    # Get predictions
    y_pred = model.predict(test_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Find misclassified examples
    misclassified_indices = np.where(y_pred_classes != y_true)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Select random misclassified examples
    np.random.shuffle(misclassified_indices)
    selected_indices = misclassified_indices[:num_examples]
    
    # Get images
    test_generator.reset()
    all_images = []
    batch_count = 0
    for images, _ in test_generator:
        all_images.extend(images)
        batch_count += len(images)
        if batch_count >= len(test_generator.filenames):
            break
    
    # Plot misclassified examples
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Misclassified Examples', fontsize=16)
    
    for i, idx in enumerate(selected_indices):
        if i >= 12:
            break
            
        row, col = i // 4, i % 4
        
        # Display image
        axes[row, col].imshow(all_images[idx])
        
        # Get prediction details
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred_classes[idx]]
        confidence = y_pred[idx][y_pred_classes[idx]]
        
        title = f"True: {true_class}\\nPred: {pred_class}\\nConf: {confidence:.3f}"
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Total misclassifications: {len(misclassified_indices)}")
    print(f"Accuracy: {1 - len(misclassified_indices)/len(y_true):.4f}")

def calculate_per_class_accuracy(y_true, y_pred_classes, class_names):
    """Calculate accuracy for each class"""
    
    accuracies = {}
    
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = np.where(y_true == i)[0]
        
        if len(class_indices) > 0:
            # Calculate accuracy for this class
            correct = np.sum(y_pred_classes[class_indices] == i)
            accuracy = correct / len(class_indices)
            accuracies[class_name] = accuracy
        else:
            accuracies[class_name] = 0.0
    
    return accuracies
'''

# Save files
with open('data_preprocessing_content.txt', 'w') as f:
    f.write(data_preprocessing_content)

with open('evaluation_content.txt', 'w') as f:
    f.write(evaluation_content)

print("✅ Data preprocessing and evaluation utilities created!")
print("Files created:")
print("- data_preprocessing_content.txt")
print("- evaluation_content.txt")