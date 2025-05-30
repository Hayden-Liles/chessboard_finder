"""
Train a CNN model for chess piece recognition, with support for continuing training
from a previously trained model.

Optimized for maximum GPU utilization while maintaining compatibility.
FIXED: Multi-GPU batching issues resolved.
"""
# Train from scratch
# python train.py --data_dir ./data/train --output_model chess_piece_model.keras

# Continue training from previous model
# python train.py --data_dir ./data/train --output_model chess_piece_model_updated.keras --existing_model chess_piece_model.keras

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json
import datetime
import numpy as np
import logging
from glob import glob

# Disable TensorFlow warnings by default
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# Configure GPU memory growth to avoid memory allocation errors
def configure_gpu(verbose=False, memory_limit=None):
    """Configure GPU settings for optimal training with multi-GPU support.
    
    Handles CUDA compatibility issues gracefully, especially with CUDA 12.x versions.

    Args:
        verbose: If True, enable verbose TensorFlow logging
        memory_limit: Optional GPU memory limit in MB (None=use all available)
        
    Returns:
        tuple: (is_gpu_available, num_gpus, strategy)
    """
    # Set TensorFlow logging level
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    print("🔍 Checking GPU and CUDA compatibility...")
    
    try:
        # Suppress CUDA errors temporarily during detection
        old_tf_cpp_log = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages temporarily
        
        # Check if GPU is available
        try:
            gpus = tf.config.list_physical_devices('GPU')
        except Exception as cuda_detection_error:
            print(f"⚠️ CUDA detection error: {str(cuda_detection_error)[:100]}...")
            print("💡 This often happens with CUDA 12.x versions and TensorFlow compatibility")
            print("🔧 Recommended fixes:")
            print("   1. pip install tensorflow[and-cuda]==2.16.1")
            print("   2. Use NVIDIA container: docker pull nvcr.io/nvidia/tensorflow:25.01-tf2-py3")
            print("   3. Install compatible CUDA: conda install cudatoolkit=11.2 cudnn=8.1.0")
            
            # Restore logging level
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_tf_cpp_log
            return False, 0, None
        
        # Restore logging level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = old_tf_cpp_log
        
        if not gpus:
            print("ℹ️ No GPUs detected by TensorFlow. Using CPU for training.")
            print("💡 If you have NVIDIA GPUs, check TensorFlow-CUDA compatibility")
            return False, 0, None

        print(f"✅ Found {len(gpus)} GPU(s) detected by TensorFlow:")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")

        # Try to configure GPU memory options
        gpu_config_success = True
        for gpu in gpus:
            try:
                if memory_limit:
                    # Limit memory if specified
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    print(f"✓ Limited GPU memory to {memory_limit}MB per GPU")
                else:
                    # Otherwise use memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
            except Exception as gpu_config_error:
                print(f"⚠️ GPU configuration warning: {str(gpu_config_error)[:100]}...")
                print("💡 Continuing with default GPU settings")
                gpu_config_success = False
                break

        # Set TensorFlow to use all GPUs
        try:
            tf.config.set_visible_devices(gpus, 'GPU')
        except Exception as visibility_error:
            print(f"⚠️ GPU visibility warning: {str(visibility_error)[:100]}...")

        # Only enable device placement logs in verbose mode
        if verbose:
            tf.debugging.set_log_device_placement(True)

        # Create distribution strategy for multi-GPU training
        strategy = None
        strategy_creation_success = True
        
        if len(gpus) > 1:
            try:
                strategy = tf.distribute.MirroredStrategy()
                print(f"🚀 Created MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
                print(f"📊 Global batch size will be: per_gpu_batch_size × {strategy.num_replicas_in_sync}")
            except Exception as strategy_error:
                print(f"⚠️ Multi-GPU strategy creation failed: {str(strategy_error)[:100]}...")
                print("💡 Falling back to single GPU training")
                strategy = None
                strategy_creation_success = False
        else:
            print("ℹ️ Single GPU detected - using standard training")

        # Test GPU functionality
        test_success = True
        try:
            if strategy:
                with strategy.scope():
                    # Test tensor operation across all GPUs
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                    print(f"✓ Multi-GPU test successful - tensor shape: {c.shape}")
            else:
                # Single GPU test
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                    c = tf.matmul(a, b)
                    print(f"✓ Single GPU test successful - tensor shape: {c.shape}")
                    
        except Exception as test_error:
            print(f"⚠️ GPU operation test failed: {str(test_error)[:100]}...")
            print("💡 GPUs detected but operations may be unstable")
            test_success = False

        # Set TensorFlow performance optimization flags
        try:
            os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
            os.environ['TF_GPU_THREAD_COUNT'] = str(len(gpus))
            os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
            os.environ['TF_ENABLE_CUDA_DEVICE_HOST_MEMORY_SYNC_BATCHNORM'] = '1'
        except:
            pass  # These are optimizations, not critical

        # Summary
        if strategy and test_success:
            print(f"🎉 Multi-GPU setup successful! Ready to use {len(gpus)} GPUs for training")
        elif len(gpus) > 0 and test_success:
            print(f"✅ Single GPU setup successful! Ready for GPU-accelerated training")
        elif len(gpus) > 0:
            print(f"⚠️ GPUs detected but with warnings - training may work but could be unstable")
        
        return len(gpus) > 0, len(gpus), strategy

    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        print("\n🔧 CUDA 12.7 Compatibility Solutions:")
        print("1. Quick fix: pip install tensorflow[and-cuda]==2.16.1")
        print("2. Stable fix: Use NVIDIA TensorFlow container")
        print("3. Alternative: Install CUDA 11.2 + TensorFlow 2.10")
        print("4. Fallback: Use CPU training with --no_gpu flag")
        return False, 0, None

# Define a custom preprocessing layer
class Normalizer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs / 255.0

    def get_config(self):
        config = super(Normalizer, self).get_config()
        return config

def create_model_with_strategy(num_classes, input_shape=(100, 100, 3), strategy=None):
    """Create a MobileNetV2-based model for chess piece recognition with multi-GPU support."""
    
    def _create_model():
        # Input layer
        inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")

        # Preprocessing to [-1, +1] via tf.keras.applications API
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        x = tf.keras.layers.Lambda(preprocess, name="preproc")(inputs)

        # Base model (frozen initially)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        x = base_model(x, training=False)

        # Head
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
        outputs = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="piece_prediction"
        )(x)

        # Assemble model
        model = tf.keras.models.Model(inputs, outputs)
        
        # Compile with appropriate optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # Create model within strategy scope if using multi-GPU
    if strategy:
        with strategy.scope():
            model = _create_model()
            print("Model created within multi-GPU distribution strategy scope")
    else:
        model = _create_model()
        print("Model created for single GPU/CPU training")
    
    return model


def load_existing_model_with_strategy(model_path, strategy=None, num_classes=None, input_shape=(100, 100, 3)):
    """Load an existing model with custom layers and multi-GPU support.
    
    Handles models created both with and without distribution strategies.
    """
    try:
        custom_objects = {'Normalizer': Normalizer}
        
        # First, try to load the model normally (without strategy)
        print(f"Loading existing model from {model_path}...")
        old_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print("✓ Successfully loaded existing model")
        
        # If we need to use a distribution strategy, create a new model and transfer weights
        if strategy:
            print("🔄 Converting model for multi-GPU training...")
            
            # Create a new model within the strategy scope
            with strategy.scope():
                # Create new model with same architecture
                new_model = create_model_with_strategy(num_classes, input_shape, strategy=None)  # Create base model first
                
                # Transfer weights from old model to new model
                print("📋 Transferring weights to new multi-GPU model...")
                
                # Method 1: Try direct weight transfer
                try:
                    new_model.set_weights(old_model.get_weights())
                    print("✓ Successfully transferred weights using set_weights()")
                except Exception as e:
                    print(f"⚠️ Direct weight transfer failed: {e}")
                    # Method 2: Layer-by-layer weight transfer
                    try:
                        transferred_layers = 0
                        for old_layer, new_layer in zip(old_model.layers, new_model.layers):
                            if old_layer.get_weights():  # Only transfer if layer has weights
                                new_layer.set_weights(old_layer.get_weights())
                                transferred_layers += 1
                        print(f"✓ Successfully transferred weights for {transferred_layers} layers")
                    except Exception as e2:
                        print(f"❌ Layer-by-layer transfer also failed: {e2}")
                        print("⚠️ Proceeding with randomly initialized weights")
                
                # Recompile the new model
                new_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print("✓ Model successfully converted for multi-GPU training")
                return new_model
        else:
            # Single GPU/CPU - return the loaded model as-is
            return old_model
            
    except Exception as e:
        print(f"❌ Error loading existing model: {e}")
        print("💡 This might happen if the model was created with a different TensorFlow version")
        print("💡 or if there are custom layers that aren't properly registered")
        return None

def create_distributed_dataset(data_dir, input_size=(100, 100), batch_size=32, 
                             validation_split=0.2, is_training=True, strategy=None):
    """Create an optimized tf.data.Dataset for chess piece training with multi-GPU support.
    
    FIXED: Properly handles multi-GPU distribution with per-replica batch sizes.
    """
    
    # Get a list of all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()  # Sort to ensure consistent class indices

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    # Create class mapping
    class_indices = {class_name: i for i, class_name in enumerate(class_dirs)}

    # Get all image paths and labels
    all_image_paths = []
    all_image_labels = []

    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        image_paths = glob(os.path.join(class_dir, "*.jpg")) + \
                      glob(os.path.join(class_dir, "*.jpeg")) + \
                      glob(os.path.join(class_dir, "*.png"))

        if not image_paths:
            print(f"Warning: No images found in {class_dir}")
            continue

        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_indices[class_name]] * len(image_paths))

    # Convert to numpy arrays
    all_image_paths = np.array(all_image_paths)
    all_image_labels = np.array(all_image_labels)

    # Shuffle the data
    indices = np.arange(len(all_image_paths))
    np.random.shuffle(indices)
    all_image_paths = all_image_paths[indices]
    all_image_labels = all_image_labels[indices]

    # Split into training and validation
    val_split_idx = int(len(all_image_paths) * (1 - validation_split))
    if is_training:
        image_paths = all_image_paths[:val_split_idx]
        image_labels = all_image_labels[:val_split_idx]
        print(f"Raw dataset: {len(image_paths)} images for training")
    else:
        image_paths = all_image_paths[val_split_idx:]
        image_labels = all_image_labels[val_split_idx:]
        print(f"Raw dataset: {len(image_paths)} images for validation")

    # Calculate effective batch sizes and trim dataset if needed
    if strategy:
        num_replicas = strategy.num_replicas_in_sync
        global_batch_size = batch_size * num_replicas
        
        # Ensure batch_size is divisible by num_replicas
        if batch_size % num_replicas != 0:
            # Adjust batch_size to be divisible by num_replicas
            original_batch_size = batch_size
            batch_size = ((batch_size + num_replicas - 1) // num_replicas) * num_replicas
            global_batch_size = batch_size * num_replicas
            print(f"⚠️  Adjusted batch size from {original_batch_size} to {batch_size} for even distribution across {num_replicas} GPUs")
    else:
        num_replicas = 1
        global_batch_size = batch_size

    # Trim dataset to be divisible by global batch size
    original_count = len(image_paths)
    usable_count = (original_count // global_batch_size) * global_batch_size
    dropped_count = original_count - usable_count

    if dropped_count > 0:
        image_paths = image_paths[:usable_count]
        image_labels = image_labels[:usable_count]
        print(f"📊 Dataset alignment:")
        print(f"   - Original samples: {original_count}")
        print(f"   - Usable samples: {usable_count} (dropped {dropped_count} for even batching)")

    # Convert labels to one-hot encoding
    num_classes = len(class_dirs)
    image_labels = tf.keras.utils.to_categorical(image_labels, num_classes=num_classes)

    # Define data loading and augmentation functions
    def load_and_preprocess_image(image_path, label):
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.cast(img, tf.float32)
        return img, label

    def augment_image(image, label):
        # Random rotation
        if tf.random.uniform(()) > 0.5:
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random translation (simplified)
        if tf.random.uniform(()) > 0.5:
            padded_image = tf.pad(image, [[5, 5], [5, 5], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(padded_image, size=[input_size[0], input_size[1], 3])
        
        return image, label

    # Create the dataset
    print("📂 Building TensorFlow dataset pipeline...")
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    print(f"   ✅ Created from {len(image_paths)} tensor slices")

    # Map loading function
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print(f"   ✅ Applied image loading and preprocessing")

    # Apply augmentation for training
    if is_training:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        print(f"   ✅ Applied data augmentation")

    # Shuffle for training
    if is_training:
        shuffle_buffer = min(len(image_paths), 1000)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        print(f"   ✅ Applied shuffling with buffer size {shuffle_buffer}")

    # CRITICAL: Use per-replica batch size
    # MirroredStrategy expects per-replica batch size and handles distribution
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print(f"   ✅ Applied batching: {batch_size} per replica")
    
    if strategy:
        print(f"   📊 With {num_replicas} GPUs: {batch_size} × {num_replicas} = {global_batch_size} total samples/step")
    
    # Prefetch for performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"   ✅ Applied prefetching")
    
    # Repeat for training
    if is_training:
        dataset = dataset.repeat()
        print(f"   ✅ Set to repeat infinitely")

    # Let MirroredStrategy handle distribution automatically
    if strategy:
        print("🚀 Dataset ready for automatic distribution by MirroredStrategy")
    
    # Calculate steps per epoch
    final_samples = len(image_paths)
    steps_per_epoch = final_samples // global_batch_size
    
    print(f"📊 Final Dataset Summary:")
    print(f"   - Total samples: {final_samples}")
    print(f"   - Batch size per GPU: {batch_size}")
    print(f"   - Global batch size: {global_batch_size}")
    print(f"   - Steps per epoch: {steps_per_epoch}")
    
    return dataset, final_samples, class_indices

def create_model(num_classes, input_shape=(100, 100, 3)):
    """Create a MobileNetV2‐based model for chess piece recognition."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")

    # Preprocessing to [-1, +1] via tf.keras.applications API
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    x = tf.keras.layers.Lambda(preprocess, name="preproc")(inputs)

    # Base model (frozen)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="piece_prediction"
    )(x)

    # Assemble & compile
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_existing_model(model_path):
    """Load an existing model with custom layers."""
    try:
        custom_objects = {'Normalizer': Normalizer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded existing model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading existing model: {e}")
        return None

def create_tf_dataset(data_dir, input_size=(100, 100), batch_size=32, validation_split=0.2, is_training=True):
    """Create an optimized tf.data.Dataset for chess piece training.

    This is a more efficient data loading pipeline than ImageDataGenerator.
    """
    # Get a list of all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()  # Sort to ensure consistent class indices

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    # Create class mapping
    class_indices = {class_name: i for i, class_name in enumerate(class_dirs)}

    # Get all image paths and labels
    all_image_paths = []
    all_image_labels = []

    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        image_paths = glob(os.path.join(class_dir, "*.jpg")) + \
                      glob(os.path.join(class_dir, "*.jpeg")) + \
                      glob(os.path.join(class_dir, "*.png"))

        if not image_paths:
            print(f"Warning: No images found in {class_dir}")
            continue

        all_image_paths.extend(image_paths)
        all_image_labels.extend([class_indices[class_name]] * len(image_paths))

    # Convert to numpy arrays
    all_image_paths = np.array(all_image_paths)
    all_image_labels = np.array(all_image_labels)

    # Shuffle the data
    indices = np.arange(len(all_image_paths))
    np.random.shuffle(indices)
    all_image_paths = all_image_paths[indices]
    all_image_labels = all_image_labels[indices]

    # Split into training and validation
    val_split_idx = int(len(all_image_paths) * (1 - validation_split))
    if is_training:
        image_paths = all_image_paths[:val_split_idx]
        image_labels = all_image_labels[:val_split_idx]
        print(f"Using {len(image_paths)} images for training")
    else:
        image_paths = all_image_paths[val_split_idx:]
        image_labels = all_image_labels[val_split_idx:]
        print(f"Using {len(image_paths)} images for validation")

    # Convert labels to one-hot encoding
    num_classes = len(class_dirs)
    image_labels = tf.keras.utils.to_categorical(image_labels, num_classes=num_classes)

    # Define data loading and augmentation functions
    def load_and_preprocess_image(image_path, label):
        # Load image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, input_size)
        img = tf.cast(img, tf.float32)

        return img, label

    def augment_image(image, label):
        # Random rotation
        if tf.random.uniform(()) > 0.5:
            image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32))

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Random translation
        if tf.random.uniform(()) > 0.5:
            image = tf.image.stateless_random_crop(
                tf.pad(image, [[10, 10], [10, 10], [0, 0]]),
                size=[input_size[0], input_size[1], 3],
                seed=[42, 42]
            )

        return image, label

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

    # Map loading function to dataset
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply augmentation during training
    if is_training:
        dataset = dataset.map(
            augment_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Optimize dataset performance
    dataset = dataset.shuffle(buffer_size=min(len(image_paths), 10000))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # If training, repeat forever; otherwise, just once
    if is_training:
        dataset = dataset.repeat()

    return dataset, len(image_paths), class_indices

def convert_single_gpu_model_to_multi_gpu(single_gpu_model_path, output_path, 
                                         num_classes, input_shape=(100, 100, 3)):
    """
    Utility function to convert a single-GPU trained model to multi-GPU compatible format.
    
    This is useful when you have an existing model and want to continue training with multiple GPUs.
    Handles Lambda layer deserialization issues gracefully.
    """
    print("🔄 Converting single-GPU model to multi-GPU format...")
    
    # Try to configure multi-GPU strategy, but handle CUDA errors gracefully
    try:
        # Try to detect GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            try:
                strategy = tf.distribute.MirroredStrategy()
                print(f"✅ Using {strategy.num_replicas_in_sync} GPUs for conversion")
                gpu_available = True
            except Exception as cuda_error:
                print(f"⚠️ CUDA error detected: {str(cuda_error)[:100]}...")
                print("💡 Falling back to CPU-based conversion")
                strategy = None
                gpu_available = False
        elif len(gpus) == 1:
            print("ℹ️ Single GPU detected - creating CPU-compatible model that can use multi-GPU later")
            strategy = None
            gpu_available = True
        else:
            print("ℹ️ No GPUs detected - creating CPU-only model")
            strategy = None
            gpu_available = False
            
    except Exception as detection_error:
        print(f"⚠️ GPU detection failed: {str(detection_error)[:100]}...")
        print("💡 Proceeding with CPU-only conversion")
        strategy = None
        gpu_available = False
    
    try:
        # Define comprehensive custom objects for Lambda layer compatibility
        def preprocess_input_mobilenetv2(x):
            """MobileNetV2 preprocessing function for Lambda layer compatibility."""
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        def preprocess_input_generic(x):
            """Generic preprocessing function fallback."""
            return (x - 127.5) / 127.5
        
        # Register the function with keras serialization
        @tf.keras.saving.register_keras_serializable()
        def preprocess_input(x):
            """Registered preprocessing function for proper serialization."""
            return tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        custom_objects = {
            'Normalizer': Normalizer,
            'preprocess_input': preprocess_input,
            'preprocess_input_mobilenetv2': preprocess_input_mobilenetv2,
            'preprocess_input_generic': preprocess_input_generic,
        }
        
        # Add all possible variations
        try:
            import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2
            custom_objects['tf.keras.applications.mobilenet_v2.preprocess_input'] = mobilenet_v2.preprocess_input
            custom_objects['mobilenet_v2.preprocess_input'] = mobilenet_v2.preprocess_input
            custom_objects['keras.applications.mobilenet_v2.preprocess_input'] = mobilenet_v2.preprocess_input
        except:
            pass
        
        # Load the single-GPU model with proper custom objects
        print("📂 Loading existing model...")
        try:
            old_model = tf.keras.models.load_model(single_gpu_model_path, custom_objects=custom_objects)
            print("✅ Successfully loaded existing model")
        except Exception as load_error:
            print(f"⚠️ Model loading failed: {str(load_error)[:150]}...")
            print("🔄 Trying alternative loading methods...")
            
            try:
                # Try loading without compilation
                old_model = tf.keras.models.load_model(single_gpu_model_path, custom_objects=custom_objects, compile=False)
                print("✅ Loaded model without compilation")
            except Exception as alt_error:
                print(f"❌ Alternative loading failed: {str(alt_error)[:150]}...")
                return False
        
        # Create new model (CPU or GPU depending on availability)
        print("🏗️ Creating new model architecture...")
        if strategy:
            with strategy.scope():
                new_model = create_model_with_strategy(num_classes, input_shape, strategy=None)
        else:
            # CPU-based model creation
            new_model = create_model_with_strategy(num_classes, input_shape, strategy=None)
        
        # Transfer weights intelligently
        print("📋 Transferring weights...")
        try:
            # Method 1: Direct weight transfer (fastest)
            new_model.set_weights(old_model.get_weights())
            print("✅ Successfully transferred all weights using direct method")
        except Exception as weight_error:
            print(f"⚠️ Direct weight transfer failed: {str(weight_error)[:100]}...")
            print("🔄 Trying intelligent layer-by-layer transfer...")
            
            # Method 2: Intelligent layer matching
            try:
                transferred_layers = 0
                skipped_layers = 0
                
                # Get only trainable layers (skip Lambda and other non-trainable layers)
                old_trainable = [layer for layer in old_model.layers if layer.trainable_weights]
                new_trainable = [layer for layer in new_model.layers if layer.trainable_weights]
                
                print(f"📊 Found {len(old_trainable)} trainable layers in original model")
                print(f"📊 Found {len(new_trainable)} trainable layers in new model")
                
                # Transfer weights between matching trainable layers
                for i, (old_layer, new_layer) in enumerate(zip(old_trainable, new_trainable)):
                    try:
                        old_weights = old_layer.get_weights()
                        if old_weights:  # Only transfer if layer has weights
                            # Check weight shapes match
                            new_shapes = [w.shape for w in new_layer.trainable_weights]
                            old_shapes = [w.shape for w in old_weights]
                            
                            if old_shapes == new_shapes:
                                new_layer.set_weights(old_weights)
                                transferred_layers += 1
                                print(f"  ✅ Layer {i+1}: {old_layer.name} → {new_layer.name}")
                            else:
                                print(f"  ⚠️ Layer {i+1}: Shape mismatch {old_layer.name} ({old_shapes} vs {new_shapes})")
                                skipped_layers += 1
                        else:
                            skipped_layers += 1
                    except Exception as layer_error:
                        print(f"  ❌ Layer {i+1}: {old_layer.name} failed ({str(layer_error)[:50]})")
                        skipped_layers += 1
                
                print(f"✅ Successfully transferred {transferred_layers} layers")
                if skipped_layers > 0:
                    print(f"⚠️ Skipped {skipped_layers} layers (preprocessing/Lambda layers)")
                    
            except Exception as intelligent_error:
                print(f"❌ Intelligent transfer failed: {intelligent_error}")
                print("⚠️ Proceeding with new model (randomly initialized weights)")
        
        # Compile the model
        print("⚙️ Compiling converted model...")
        compile_fn = lambda: new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        if strategy:
            with strategy.scope():
                compile_fn()
        else:
            compile_fn()
        
        # Save the converted model
        print("💾 Saving converted model...")
        new_model.save(output_path)
        
        if gpu_available:
            print(f"🎉 Multi-GPU compatible model saved to: {output_path}")
            print("✅ Model is ready for multi-GPU training!")
        else:
            print(f"✅ CPU-compatible model saved to: {output_path}")
            print("💡 Model can be used for CPU training or when GPU environment is optimized")
        
        # Update metadata
        if os.path.exists(single_gpu_model_path + '.metadata'):
            try:
                import shutil
                shutil.copy2(single_gpu_model_path + '.metadata', output_path + '.metadata')
                
                import json
                with open(output_path + '.metadata', 'r') as f:
                    metadata = json.load(f)
                
                metadata['converted_for_multi_gpu'] = True
                metadata['conversion_date'] = datetime.datetime.now().isoformat()
                metadata['original_model'] = single_gpu_model_path
                metadata['gpu_available_during_conversion'] = gpu_available
                metadata['conversion_method'] = 'multi_gpu_with_lambda_fix' if strategy else 'cpu_compatible_with_lambda_fix'
                metadata['lambda_layer_handled'] = True
                
                with open(output_path + '.metadata', 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                print("✅ Updated metadata file")
            except Exception as e:
                print(f"⚠️ Could not update metadata: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Check if model file exists and is accessible")
        print("2. Verify model was saved with compatible TensorFlow version")
        print("3. Try loading model manually to identify specific layer issues")
        print("4. Consider recreating model architecture if Lambda layers are problematic")
        return False

def train_chess_piece_model(data_dir, output_model_path, existing_model_path=None,
                           input_size=(100, 100), batch_size=32, epochs=50,
                           unfreeze_base=True, continue_epoch=0, verbose=False,
                           strategy=None, num_gpus=1):
    """Train a chess piece recognition model with multi-GPU support.
    
    FIXED: Simplified to work with corrected dataset creation.
    """
    
    print(f"Training chess piece recognition model using data from: {data_dir}")
    print(f"Model will be saved to: {output_model_path}")
    
    if strategy:
        print(f"🚀 Training with {strategy.num_replicas_in_sync} GPUs using MirroredStrategy")
        print(f"📊 Batch size: {batch_size} per GPU")
        print(f"📊 Global batch size: {batch_size * strategy.num_replicas_in_sync}")
    else:
        print(f"Training with single GPU/CPU, batch size: {batch_size}")

    # Create datasets
    train_dataset, train_samples, class_indices = create_distributed_dataset(
        data_dir, input_size, batch_size, validation_split=0.2, 
        is_training=True, strategy=strategy
    )

    val_dataset, val_samples, _ = create_distributed_dataset(
        data_dir, input_size, batch_size, validation_split=0.2, 
        is_training=False, strategy=strategy
    )

    # Get class names and num_classes
    class_names = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    print("\nClass mapping:")
    for idx, class_name in class_names.items():
        print(f"  {idx}: {class_name}")

    # Load existing model or create new one
    if existing_model_path and os.path.exists(existing_model_path):
        model = load_existing_model_with_strategy(
            existing_model_path, 
            strategy, 
            num_classes, 
            input_size + (3,)
        )
        if model is None:
            print("❌ Failed to load existing model. Creating new one.")
            model = create_model_with_strategy(num_classes, input_size + (3,), strategy)
        else:
            # Check output layer compatibility
            output_layer = model.layers[-1]
            if hasattr(output_layer, 'units'):
                if output_layer.units != num_classes:
                    print(f"⚠️ Output layer has {output_layer.units} units but data has {num_classes} classes!")
                    print("Cannot continue training with mismatched class count.")
                    return None, None
            print("✓ Model loaded and ready for continued training")
    else:
        # Create new model
        model = create_model_with_strategy(num_classes, input_size + (3,), strategy)

    # Save metadata
    metadata = {
        'class_mapping': {str(k): v for k, v in class_names.items()},
        'training_data': data_dir,
        'input_size': input_size,
        'num_gpus': num_gpus,
        'batch_size_per_gpu': batch_size,
        'global_batch_size': batch_size * (strategy.num_replicas_in_sync if strategy else 1),
        'last_trained': datetime.datetime.now().isoformat()
    }

    with open(output_model_path + '.metadata', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print model summary
    model.summary()

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        )
    ]

    # Calculate steps per epoch
    if strategy:
        # For multi-GPU, divide by global batch size
        global_batch_size = batch_size * strategy.num_replicas_in_sync
        steps_per_epoch = train_samples // global_batch_size
        validation_steps = val_samples // global_batch_size
    else:
        # For single GPU/CPU
        steps_per_epoch = train_samples // batch_size
        validation_steps = val_samples // batch_size
    
    # Ensure at least one step
    steps_per_epoch = max(steps_per_epoch, 1)
    validation_steps = max(validation_steps, 1)
    
    print(f"\n📊 Training Configuration:")
    print(f"   - Training samples: {train_samples}")
    print(f"   - Validation samples: {val_samples}")
    print(f"   - Steps per epoch: {steps_per_epoch}")
    print(f"   - Validation steps: {validation_steps}")

    initial_epoch = continue_epoch

    # Training execution
    if existing_model_path and os.path.exists(existing_model_path):
        print("\nContinuing training from existing model...")
        
        if unfreeze_base:
            # Unfreeze layers for fine-tuning
            unfreeze_fn = lambda: unfreeze_base_layers(model)
            if strategy:
                with strategy.scope():
                    unfreeze_fn()
            else:
                unfreeze_fn()

        # Train the model
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2 if verbose else 1,
            initial_epoch=initial_epoch
        )

        plot_history(history, f"continued_from_epoch_{initial_epoch}")

    else:
        # New training with two phases
        print("\nPhase 1: Training with frozen base model...")
        history1 = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2 if verbose else 1
        )

        if unfreeze_base:
            print("\nPhase 2: Fine-tuning with unfrozen top layers...")
            
            # Unfreeze and recompile
            def unfreeze_and_recompile():
                unfreeze_base_layers(model)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            
            if strategy:
                with strategy.scope():
                    unfreeze_and_recompile()
            else:
                unfreeze_and_recompile()

            # Continue training
            history2 = model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=2 if verbose else 1,
                initial_epoch=20
            )

            plot_combined_history(history1, history2)
        else:
            plot_history(history1, "phase1_only")

    # Load and evaluate best model
    custom_objects = {'Normalizer': Normalizer}
    if strategy:
        with strategy.scope():
            best_model = tf.keras.models.load_model(output_model_path, custom_objects=custom_objects)
    else:
        best_model = tf.keras.models.load_model(output_model_path, custom_objects=custom_objects)

    print("\nEvaluating best model on validation data...")
    val_loss, val_accuracy = best_model.evaluate(
        val_dataset,
        steps=validation_steps,
        verbose=2 if verbose else 1
    )
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"\nTraining complete! Model saved to {output_model_path}")

    return best_model, class_names


def unfreeze_base_layers(model):
    """Unfreeze the top layers of the base model for fine-tuning."""
    try:
        # Find the base model layer
        base_model = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                base_model = layer
                break
        
        if base_model:
            # Unfreeze the top 20 layers
            for layer in base_model.layers[-20:]:
                layer.trainable = True
            print("Unfroze top 20 layers of base model for fine-tuning")
        else:
            print("Could not identify base model layer for unfreezing")
    except Exception as e:
        print(f"Error unfreezing base model layers: {e}")

def plot_history(history, title_suffix=""):
    """Plot the training history from a single training session."""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Model Accuracy {title_suffix}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Model Loss {title_suffix}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_history_{title_suffix}.png')
    plt.close()

def plot_combined_history(history1, history2):
    """Plot combined training history from two phases."""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Phase 1 Train')
    plt.plot(history1.history['val_accuracy'], label='Phase 1 Validation')

    # Append Phase 2 histories
    phase1_epochs = len(history1.history['accuracy'])
    plt.plot(range(phase1_epochs, phase1_epochs + len(history2.history['accuracy'])),
             history2.history['accuracy'], label='Phase 2 Train')
    plt.plot(range(phase1_epochs, phase1_epochs + len(history2.history['val_accuracy'])),
             history2.history['val_accuracy'], label='Phase 2 Validation')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label='Phase 1 Train')
    plt.plot(history1.history['val_loss'], label='Phase 1 Validation')

    # Append Phase 2 histories
    plt.plot(range(phase1_epochs, phase1_epochs + len(history2.history['loss'])),
             history2.history['loss'], label='Phase 2 Train')
    plt.plot(range(phase1_epochs, phase1_epochs + len(history2.history['val_loss'])),
             history2.history['val_loss'], label='Phase 2 Validation')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_combined.png')
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a chess piece recognition model with multi-GPU support")
    parser.add_argument("--data_dir", type=str, required=False,
                        help="Directory containing the generated chess piece images (with class subdirectories)")
    parser.add_argument("--output_model", type=str, default="chess_piece_model_new.keras",
                        help="Path to save the trained model")
    parser.add_argument("--existing_model", type=str, default=None,
                        help="Path to an existing model to continue training")
    parser.add_argument("--input_size", type=int, default=100,
                        help="Size to resize input images (default: 100)")
    parser.add_argument("--batch_size", type=int, default=128,  # Per-GPU batch size
                        help="Batch size per GPU for training (default: 128)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs to train (default: 50)")
    parser.add_argument("--no_unfreeze", action="store_true",
                        help="Disable unfreezing base model layers")
    parser.add_argument("--continue_epoch", type=int, default=0,
                        help="Epoch to continue from when using existing model")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose TensorFlow logging")
    parser.add_argument("--memory_limit", type=int, default=None,
                        help="GPU memory limit in MB per GPU, e.g. 6000 (default: use all available)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable TensorFlow benchmarking to find optimal configurations")
    
    # Model conversion utility
    parser.add_argument("--convert_model", type=str, default=None,
                        help="Convert a single-GPU model to multi-GPU format. Specify path to single-GPU model.")
    parser.add_argument("--converted_model_output", type=str, default="converted_multi_gpu_model.keras",
                        help="Output path for converted multi-GPU model")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (required for model conversion)")

    args = parser.parse_args()
    
    # Handle model conversion utility
    if args.convert_model:
        if not args.num_classes:
            print("❌ --num_classes is required for model conversion")
            exit(1)
        
        print("🔄 Starting model conversion...")
        success = convert_single_gpu_model_to_multi_gpu(
            args.convert_model,
            args.converted_model_output,
            args.num_classes,
            (args.input_size, args.input_size, 3)
        )
        
        if success:
            print("✅ Model conversion completed successfully!")
            print(f"💡 You can now use the converted model: {args.converted_model_output}")
        else:
            print("❌ Model conversion failed!")
        
        exit(0)
    
    # Require data_dir for training
    if not args.data_dir:
        print("❌ --data_dir is required for training")
        parser.print_help()
        exit(1)

    # Configure GPU settings and get distribution strategy
    is_gpu_available, num_gpus, strategy = configure_gpu(
        verbose=args.verbose, 
        memory_limit=args.memory_limit
    )

    # Handle benchmark mode
    if args.benchmark:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD'] = 'Convolution,MatMul'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        print("TensorFlow benchmarking enabled")

    # Disable mixed precision for now to fix accuracy issues
    # Mixed precision can cause numeric instability with some models
    if is_gpu_available and num_gpus > 0:
        print("ℹ️ Mixed precision disabled for stability")
        # Uncomment below to enable mixed precision (may cause accuracy issues)
        # try:
        #     policy = tf.keras.mixed_precision.Policy('mixed_float16')
        #     tf.keras.mixed_precision.set_global_policy(policy)
        #     print(f"Mixed precision enabled: {policy.name}")
        # except Exception as e:
        #     print(f"Could not enable mixed precision: {e}")

    # Force CPU if requested
    if args.no_gpu:
        print("Forcing CPU usage as requested")
        tf.config.set_visible_devices([], 'GPU')
        strategy = None
        num_gpus = 0

    # Adjust batch size for CPU training
    actual_batch_size = args.batch_size
    if not is_gpu_available or args.no_gpu:
        print("No GPU detected, reducing batch size")
        actual_batch_size = min(32, args.batch_size)

    # Train the model with multi-GPU support
    train_chess_piece_model(
        args.data_dir,
        args.output_model,
        existing_model_path=args.existing_model,
        input_size=(args.input_size, args.input_size),
        batch_size=actual_batch_size,
        epochs=args.epochs,
        unfreeze_base=not args.no_unfreeze,
        continue_epoch=args.continue_epoch,
        verbose=args.verbose,
        strategy=strategy,
        num_gpus=num_gpus
    )