import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_transfer_model(input_shape=(224, 224, 3), num_classes=5, base_arch='resnet50'):
    if base_arch == 'resnet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_arch == 'mobilenet':
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_arch == 'vgg16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Architecture {base_arch} not supported")

    base_model.trainable = True # Fine-tuning

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def build_crnn_model(input_shape=(128, 431, 1), num_classes=5):
    # Input: (Mel-bins, Time-frames, Channels)
    inputs = layers.Input(shape=input_shape)
    
    # CNN Block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Reshape for RNN: (Batch, Time, Features)
    # New shape calculation needed based on pooling
    # Assuming input (128, 431, 1) -> (16, 53, 128)
    target_shape = (x.shape[2], x.shape[1] * x.shape[3]) 
    x = layers.Reshape(target_shape)(x)
    
    # RNN Block
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    
    # Classifier
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def build_attention_model(input_shape=(224, 224, 3), num_classes=5):
    """CNN with attention mechanism for Keras/TensorFlow."""
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    # Attention mechanism
    attn = layers.Conv2D(128, (1, 1), activation='relu')(x)
    attn = layers.Conv2D(1, (1, 1), activation='sigmoid')(attn)
    
    # Apply attention
    x = layers.Multiply()([x, attn])
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def build_ensemble_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Ensemble of multiple architectures for robust predictions.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Branch 1: ResNet
    resnet_base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet_base.trainable = False
    x1 = resnet_base(inputs)
    x1 = layers.GlobalAveragePooling2D()(x1)
    
    # Branch 2: MobileNet
    mobile_base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    mobile_base.trainable = False
    x2 = mobile_base(inputs)
    x2 = layers.GlobalAveragePooling2D()(x2)
    
    # Concatenate branches
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

def compile_model(model, learning_rate=0.001, loss='categorical_crossentropy'):
    """
    Compiles a Keras model with standard optimizer and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model
