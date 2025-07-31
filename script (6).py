# Create model architecture file
model_architecture_content = '''
"""
Enhanced Model Architecture for Plant Leaf Classification
Combines transfer learning with custom layers for high accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config.config import Config

def create_base_model(model_name='EfficientNetB3'):
    """Create base model using transfer learning"""
    
    if model_name == 'EfficientNetB3':
        base_model = tf.keras.applications.EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3),
            drop_connect_rate=0.4
        )
    elif model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
        )
    elif model_name == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Freeze initial layers, unfreeze last few layers for fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    return base_model

def attention_block(inputs, filters):
    """Attention mechanism for feature enhancement"""
    
    # Channel attention
    channel_avg = layers.GlobalAveragePooling2D()(inputs)
    channel_max = layers.GlobalMaxPooling2D()(inputs)
    
    channel_avg = layers.Dense(filters // 8, activation='relu')(channel_avg)
    channel_avg = layers.Dense(filters, activation='sigmoid')(channel_avg)
    
    channel_max = layers.Dense(filters // 8, activation='relu')(channel_max)
    channel_max = layers.Dense(filters, activation='sigmoid')(channel_max)
    
    channel_attention = layers.Add()([channel_avg, channel_max])
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    
    # Apply channel attention
    attended_features = layers.Multiply()([inputs, channel_attention])
    
    return attended_features

def create_enhanced_model(num_classes):
    """Create enhanced model with attention and multiple techniques"""
    
    # Input layer
    inputs = layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))
    
    # Data augmentation layers (applied during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # Normalization
    x = layers.Rescaling(1./255)(x)
    
    # Base model (EfficientNet)
    base_model = create_base_model('EfficientNetB3')
    x = base_model(x, training=False)
    
    # Add attention mechanism
    x = attention_block(x, x.shape[-1])
    
    # Global pooling with both average and max
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    # Concatenate pooled features
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)
    
    # Dense layers with residual connections
    x1 = layers.Dense(512, activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    x2 = layers.Dense(256, activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)
    
    # Output layer
    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x2)
    
    # Create model
    model = tf.keras.Model(inputs, predictions)
    
    return model

def create_ensemble_model(num_classes):
    """Create ensemble model combining multiple architectures"""
    
    # Input layer
    inputs = layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1./255)(x)
    
    # Create multiple base models
    efficientnet = create_base_model('EfficientNetB3')
    resnet = create_base_model('ResNet50')
    densenet = create_base_model('DenseNet121')
    
    # Get features from each model
    eff_features = efficientnet(x)
    res_features = resnet(x)
    dense_features = densenet(x)
    
    # Global pooling for each
    eff_pool = layers.GlobalAveragePooling2D()(eff_features)
    res_pool = layers.GlobalAveragePooling2D()(res_features)
    dense_pool = layers.GlobalAveragePooling2D()(dense_features)
    
    # Concatenate all features
    combined = layers.Concatenate()([eff_pool, res_pool, dense_pool])
    
    # Dense layers
    x = layers.Dense(1024, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    ensemble_model = tf.keras.Model(inputs, predictions)
    
    return ensemble_model

def create_lightweight_model(num_classes):
    """Create lightweight model for mobile deployment"""
    
    inputs = layers.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))
    
    # Normalization
    x = layers.Rescaling(1./255)(inputs)
    
    # MobileNetV2 as base
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3),
        alpha=1.0
    )
    base_model.trainable = True
    
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, predictions)
    
    return model
'''

# Create config file
config_content = '''
"""
Configuration file for Plant Leaf Classification System
"""

class Config:
    # Data configuration
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    
    # Training configuration
    EPOCHS = 100
    INITIAL_LEARNING_RATE = 0.001
    RANDOM_SEED = 42
    
    # Model configuration
    EARLY_STOPPING_PATIENCE = 15
    
    # Data paths
    DATA_PATH = "data/"
    MODEL_SAVE_PATH = "saved_models/"
    
    # Data split ratios
    TRAIN_SPLIT = 0.7
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Augmentation parameters
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = False
    BRIGHTNESS_RANGE = [0.8, 1.2]
    
    # Class names for PlantVillage dataset
    CLASS_NAMES = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
'''

# Save files
with open('model_architecture_content.txt', 'w') as f:
    f.write(model_architecture_content)

with open('config_content.txt', 'w') as f:
    f.write(config_content)

print("✅ Model architecture and config files created!")
print("Files created:")
print("- model_architecture_content.txt")
print("- config_content.txt")