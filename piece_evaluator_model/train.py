"""
Train a CNN model for chess piece recognition, with support for continuing training
from a previously trained model.
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import datetime

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
    """Create a MobileNetV2-based model for chess piece recognition."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")
    
    # Add preprocessing (normalization) as part of the model
    x = Normalizer(name="normalizer")(inputs)
    
    # Base model - MobileNetV2 for feature extraction
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Don't include the classification head
        weights='imagenet',  # Use ImageNet pre-trained weights
    )
    
    # Freeze the base model layers to start with
    base_model.trainable = False
    
    # Apply the base model to input
    x = base_model(x)
    
    # Add classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="piece_prediction")(x)
    
    # Create the complete model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
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

def train_chess_piece_model(data_dir, output_model_path, existing_model_path=None, 
                           input_size=(100, 100), batch_size=32, epochs=50, 
                           unfreeze_base=True, continue_epoch=0):
    """Train a chess piece recognition model, optionally continuing from an existing model."""
    print(f"Training chess piece recognition model using data from: {data_dir}")
    print(f"Model will be saved to: {output_model_path}")
    
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
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        validation_split=0.2,  # 20% for validation
        rotation_range=10,     # Rotate images slightly
        width_shift_range=0.1, # Shift horizontally
        height_shift_range=0.1,# Shift vertically
        zoom_range=0.1,        # Zoom in/out slightly
        brightness_range=[0.9, 1.1],  # Adjust brightness
        horizontal_flip=False, # Don't flip chess pieces (could confuse bishop/knight/etc)
        fill_mode='nearest'    # Fill in missing pixels after transformations
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get current class indices and names
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
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
    
    # Print class mapping
    print("\nClass mapping:")
    for idx, class_name in class_names.items():
        print(f"  {idx}: {class_name}")
    
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
    
    # Create or load the model
    num_classes = len(class_indices)
    if existing_model_path and os.path.exists(existing_model_path):
        model = load_existing_model(existing_model_path)
        if model is None:
            print("Failed to load existing model. Creating new one.")
            model = create_model(num_classes, input_size + (3,))
        else:
            # Check if output layer matches the number of classes
            output_layer = model.layers[-1]
            if output_layer.output_shape[-1] != num_classes:
                print(f"⚠️ Output layer has {output_layer.output_shape[-1]} units but data has {num_classes} classes!")
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
    
    # Print model summary
    model.summary()
    
    # Set up callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=output_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stop early if no improvement
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    
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
        
        # Train with all training data
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=initial_epoch
        )
        
        # Plot training history
        plot_history(history, f"continued_from_epoch_{initial_epoch}")
        
    else:
        # New training with two phases
        print("\nPhase 1: Training with frozen base model...")
        history1 = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
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
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,  # Continue for all epochs
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
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
    val_loss, val_accuracy = best_model.evaluate(validation_generator, steps=validation_steps)
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
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of epochs to train (default: 50)")
    parser.add_argument("--no_unfreeze", action="store_true",
                        help="Disable unfreezing base model layers")
    parser.add_argument("--continue_epoch", type=int, default=0,
                        help="Epoch to continue from when using existing model")
    
    args = parser.parse_args()
    
    # Train the model
    train_chess_piece_model(
        args.data_dir,
        args.output_model,
        existing_model_path=args.existing_model,
        input_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        unfreeze_base=not args.no_unfreeze,
        continue_epoch=args.continue_epoch
    )