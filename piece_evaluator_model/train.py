"""
Train a CNN model for chess piece recognition, with support for continuing training
from a previously trained model.

Optimized for maximum GPU utilization while maintaining compatibility.
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
    """Configure GPU settings for optimal training.

    Args:
        verbose: If True, enable verbose TensorFlow logging
        memory_limit: Optional GPU memory limit in MB (None=use all available)
    """
    # Set TensorFlow logging level
    if not verbose:
        # Reduce TensorFlow logging verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU detected. Using CPU for training.")
            return False

        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu.name}")

        # Configure GPU memory options
        for gpu in gpus:
            if memory_limit:
                # Limit memory if specified
                try:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    print(f"Limited GPU memory to {memory_limit}MB")
                except Exception as e:
                    print(f"Error setting memory limit: {e}")
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                # Otherwise use memory growth
                tf.config.experimental.set_memory_growth(gpu, True)

        # Set TensorFlow to use the GPU
        tf.config.set_visible_devices(gpus, 'GPU')

        # Only enable device placement logs in verbose mode
        tf.debugging.set_log_device_placement(verbose)

        # Create a small test tensor on GPU to verify
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print(f"Test tensor operation result shape: {c.shape}")

        # Set TensorFlow performance optimization flags
        # These can help with GPU utilization
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
        os.environ['TF_GPU_THREAD_COUNT'] = '1'
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
        os.environ['TF_ENABLE_CUDA_DEVICE_HOST_MEMORY_SYNC_BATCHNORM'] = '1'

        return True
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        print("Falling back to CPU.")
        return False

# Define a custom preprocessing layer
class Normalizer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs / 255.0

    def get_config(self):
        config = super(Normalizer, self).get_config()
        return config

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

def train_chess_piece_model(data_dir, output_model_path, existing_model_path=None,
                           input_size=(100, 100), batch_size=32, epochs=50,
                           unfreeze_base=True, continue_epoch=0, verbose=False):
    """Train a chess piece recognition model, optionally continuing from an existing model."""
    print(f"Training chess piece recognition model using data from: {data_dir}")
    print(f"Model will be saved to: {output_model_path}")

    # First, create the datasets to know how many classes we have
    # Create optimized datasets using tf.data API
    train_dataset, train_samples, class_indices = create_tf_dataset(
        data_dir, input_size, batch_size, validation_split=0.2, is_training=True
    )

    val_dataset, val_samples, _ = create_tf_dataset(
        data_dir, input_size, batch_size, validation_split=0.2, is_training=False
    )

    # Get class names in correct order and define num_classes
    class_names = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)

    # Print class mapping
    print("\nClass mapping:")
    for idx, class_name in class_names.items():
        print(f"  {idx}: {class_name}")

    # Now load existing model if specified
    if existing_model_path and os.path.exists(existing_model_path):
        model = load_existing_model(existing_model_path)
        if model is None:
            print("Failed to load existing model. Creating new one.")
            model = create_model(num_classes, input_size + (3,))
        else:
            # Check if output layer matches the number of classes
            output_layer = model.layers[-1]
            # Use units attribute instead of output_shape
            if hasattr(output_layer, 'units'):
                if output_layer.units != num_classes:
                    print(f"⚠️ Output layer has {output_layer.units} units but data has {num_classes} classes!")
                    print("Cannot continue training with mismatched class count.")
                    return None, None
            else:
                # Fallback to output_shape for older Keras versions
                if output_layer.output_shape[-1] != num_classes:
                    print(f"⚠️ Output layer has {output_layer.output_shape[-1]} outputs but data has {num_classes} classes!")
                    print("Cannot continue training with mismatched class count.")
                    return None, None

            # Recompile with fresh optimizer for continued training
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    else:
        # Create new model
        model = create_model(num_classes, input_size + (3,))

    # Load metadata from previous training if continuing
    previous_class_mapping = None
    if existing_model_path and os.path.exists(existing_model_path + '.metadata'):
        try:
            with open(existing_model_path + '.metadata', 'r') as f:
                metadata = json.load(f)
                previous_class_mapping = metadata.get('class_mapping', {})
                print(f"Loaded previous class mapping with {len(previous_class_mapping)} classes")
        except Exception as e:
            print(f"Warning: Could not load previous metadata: {e}")

    # Check class consistency if continuing from an existing model
    if previous_class_mapping:
        mismatch = False
        for idx, name in previous_class_mapping.items():
            if int(idx) < len(class_names) and class_names[int(idx)] != name:
                print(f"⚠️ Class index mismatch: Previous {idx}: {name}, Current {idx}: {class_names[int(idx)]}")
                mismatch = True

        if mismatch:
            print("WARNING: Class mapping has changed between training sessions!")
            print("This may cause incorrect predictions. Consider:")
            print("1. Organizing your data directories to maintain the same class order")
            print("2. Starting fresh with a new model instead of continuing training")
            user_input = input("Continue anyway? (y/n): ")
            if user_input.lower() != 'y':
                print("Training aborted.")
                return None, None

    # Save class mapping to a metadata file
    metadata = {
        'class_mapping': {str(k): v for k, v in class_names.items()},
        'training_data': data_dir,
        'input_size': input_size,
        'last_trained': datetime.datetime.now().isoformat()
    }

    with open(output_model_path + '.metadata', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save as readable text file
    with open('class_mapping.txt', 'w') as f:
        f.write("Class index to name mapping:\n")
        for idx, class_name in class_names.items():
            f.write(f"{idx}: {class_name}\n")

    # Print model summary
    model.summary()

    # Set up callbacks
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stop early if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0  # Set to a range like '1,10' to profile those batches
        )
    ]

    # Calculate steps per epoch
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size

    # Ensure at least one step in validation
    validation_steps = max(validation_steps, 1)

    # If continuing training, we can go directly to fine-tuning
    initial_epoch = continue_epoch

    # Training strategy
    if existing_model_path and os.path.exists(existing_model_path):
        print("\nContinuing training from existing model...")

        if unfreeze_base:
            # Optionally unfreeze some layers for fine-tuning
            try:
                # Find the base model - name might vary
                base_model = None
                for layer in model.layers:
                    if 'mobilenetv2' in layer.name.lower():
                        base_model = layer
                        break

                if base_model:
                    # Unfreeze the top layers
                    for layer in base_model.layers[-20:]:
                        layer.trainable = True
                    print("Unfroze top 20 layers of base model for fine-tuning")
                else:
                    print("Could not identify base model layer for unfreezing")
            except Exception as e:
                print(f"Error unfreezing base model layers: {e}")

        # Train with all training data using efficient tf.data API
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=2 if verbose else 1,  # 2=one line per epoch, 1=progress bar
            initial_epoch=initial_epoch
        )

        # Plot training history
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
            verbose=2 if verbose else 1  # 2=one line per epoch, 1=progress bar
        )

        if unfreeze_base:
            # Fine-tune by unfreezing some layers
            print("\nPhase 2: Fine-tuning with unfrozen top layers...")

            # Unfreeze the top layers of the base model
            try:
                base_model = model.get_layer('mobilenetv2_1.00_100')
                # Unfreeze the last 20 layers
                for layer in base_model.layers[-20:]:
                    layer.trainable = True
            except:
                # Try with different layer name pattern
                for layer in model.layers:
                    if 'mobilenetv2' in layer.name.lower():
                        base_model = layer
                        for sublayer in base_model.layers[-20:]:
                            sublayer.trainable = True
                        break

            # Re-compile the model with a lower learning rate
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            # Continue training with unfrozen layers
            history2 = model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,  # Continue for all epochs
                validation_data=val_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=2 if verbose else 1,  # 2=one line per epoch, 1=progress bar
                initial_epoch=20  # Start from epoch 20
            )

            # Plot combined training history
            plot_combined_history(history1, history2)
        else:
            # Plot just the first phase history
            plot_history(history1, "phase1_only")

    # Load the best model
    best_model = tf.keras.models.load_model(
        output_model_path,
        custom_objects={'Normalizer': Normalizer}
    )

    # Evaluate the best model
    print("\nEvaluating best model on validation data...")
    val_loss, val_accuracy = best_model.evaluate(
        val_dataset,
        steps=validation_steps,
        verbose=2 if verbose else 1  # 2=one line per epoch, 1=progress bar
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print(f"\nTraining complete! Model saved to {output_model_path}")

    return best_model, class_names

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

    parser = argparse.ArgumentParser(description="Train a chess piece recognition model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the generated chess piece images (with class subdirectories)")
    parser.add_argument("--output_model", type=str, default="chess_piece_model_new.keras",
                        help="Path to save the trained model")
    parser.add_argument("--existing_model", type=str, default=None,
                        help="Path to an existing model to continue training")
    parser.add_argument("--input_size", type=int, default=100,
                        help="Size to resize input images (default: 100)")
    parser.add_argument("--batch_size", type=int, default=512,  # Significantly increased for better GPU utilization
                        help="Batch size for training (default: 512)")
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
                        help="GPU memory limit in MB, e.g. 6000 (default: use all available)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Enable TensorFlow benchmarking to find optimal configurations")

    args = parser.parse_args()

    # Configure GPU settings
    if args.benchmark:
        # Enable TensorFlow benchmarking
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD'] = 'Convolution,MatMul'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        print("TensorFlow benchmarking enabled")

    # Configure GPU at startup
    is_gpu_available = configure_gpu(verbose=args.verbose, memory_limit=args.memory_limit)

    # Optionally enable mixed precision for faster training on compatible GPUs
    if is_gpu_available:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Mixed precision enabled: {policy.name}")
        except Exception as e:
            print(f"Could not enable mixed precision: {e}")
            print("Training will continue with default precision")

    # Force CPU if requested
    if args.no_gpu:
        print("Forcing CPU usage as requested")
        tf.config.set_visible_devices([], 'GPU')

    # Adjusted batch size based on GPU availability
    actual_batch_size = args.batch_size
    if not is_gpu_available:
        print("No GPU detected, reducing batch size")
        actual_batch_size = min(32, args.batch_size)

    # Train the model
    train_chess_piece_model(
        args.data_dir,
        args.output_model,
        existing_model_path=args.existing_model,
        input_size=(args.input_size, args.input_size),
        batch_size=actual_batch_size,
        epochs=args.epochs,
        unfreeze_base=not args.no_unfreeze,
        continue_epoch=args.continue_epoch,
        verbose=args.verbose
    )
