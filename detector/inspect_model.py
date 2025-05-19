"""
Model inspection script for TensorFlow models with custom layers.

This script attempts to load a TensorFlow model and print its architecture,
which can help diagnose issues with custom layers or operations.
"""
import os
import sys
import tensorflow as tf
import numpy as np

def inspect_model(model_path):
    """Attempt to load a model and print its structure."""
    print(f"Attempting to load model: {model_path}")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        return
    
    # Try different loading approaches
    try:
        # Approach 1: Load with compile=False
        print("\nApproach 1: Loading with compile=False")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Success! Model loaded.")
        print_model_info(model)
    except Exception as e:
        print(f"Approach 1 failed: {e}")
    
    try:
        # Approach 2: Custom objects with TrueDivide
        print("\nApproach 2: Using custom_object_scope with TrueDivide")
        custom_objects = {
            'TrueDivide': tf.keras.layers.Lambda(lambda x: x / 255.0)
        }
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)
        print("Success! Model loaded with custom objects.")
        print_model_info(model)
    except Exception as e:
        print(f"Approach 2 failed: {e}")
    
    try:
        # Approach 3: Load as h5py file to inspect structure
        print("\nApproach 3: Loading as h5py file")
        import h5py
        with h5py.File(model_path, 'r') as f:
            print("Model file opened as HDF5")
            print("Root groups:", list(f.keys()))
            if 'model_weights' in f:
                print("Model weights group exists")
                print("Weight layers:", list(f['model_weights'].keys()))
    except Exception as e:
        print(f"Approach 3 failed: {e}")
    
    try:
        # Approach 4: Using SavedModel loader
        print("\nApproach 4: Check if it's a SavedModel directory")
        if os.path.isdir(model_path):
            print(f"{model_path} is a directory, checking if it's a SavedModel")
            model = tf.saved_model.load(model_path)
            print("Success! Loaded as SavedModel")
            print("Available signatures:", list(model.signatures.keys()))
        else:
            print(f"{model_path} is not a directory, skipping SavedModel check")
    except Exception as e:
        print(f"Approach 4 failed: {e}")
    
    print("\nInspection complete")

def print_model_info(model):
    """Print information about a loaded model."""
    print("\nModel Summary:")
    model.summary()
    
    print("\nModel Layers:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}, Type: {type(layer).__name__}")
    
    print("\nInput shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    
    # Check if it has a predict function
    if hasattr(model, 'predict'):
        print("\nModel has predict function")
        # Create a random input and test prediction
        try:
            input_shape = model.input_shape[1:]
            test_input = np.random.random((1,) + input_shape)
            print(f"Testing prediction with random input of shape {test_input.shape}")
            prediction = model.predict(test_input, verbose=0)
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction values (first 5): {prediction[0][:5]}")
            print(f"Predicted class: {np.argmax(prediction[0])}")
        except Exception as e:
            print(f"Error during prediction test: {e}")
    else:
        print("\nModel does not have a predict function")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py path/to/model.h5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_model(model_path)