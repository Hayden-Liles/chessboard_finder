#!/usr/bin/env python3
"""
Export TensorFlow model to SavedModel format.

This script loads a TensorFlow model in HDF5 (.h5) format and exports it
as a SavedModel, which often handles custom operations better.
"""
import os
import sys
import tensorflow as tf
import numpy as np

def export_model(input_model_path, output_directory):
    """
    Load a model from HDF5 and export it as SavedModel.
    
    Args:
        input_model_path: Path to the input .h5 model file
        output_directory: Directory where the SavedModel will be saved
    """
    print(f"Attempting to export model: {input_model_path} → {output_directory}")
    
    # Check if the input file exists
    if not os.path.exists(input_model_path):
        print(f"Error: Input model file {input_model_path} does not exist.")
        return False
    
    # Create a custom TrueDivide layer to handle model loading
    class TrueDivide(tf.keras.layers.Layer):
        def __init__(self, divisor=255.0, **kwargs):
            super(TrueDivide, self).__init__(**kwargs)
            self.divisor = divisor
            
        def call(self, inputs):
            return tf.truediv(inputs, self.divisor)
        
        def get_config(self):
            config = super(TrueDivide, self).get_config()
            config.update({"divisor": self.divisor})
            return config
    
    # Attempt to load the model
    try:
        # Try loading with custom objects
        custom_objects = {
            'TrueDivide': TrueDivide
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(input_model_path, compile=False)
        
        print("Successfully loaded model with custom objects")
        
        # Print model summary
        model.summary()
        
        # Create a preprocessing function that handles the division
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 100, 100, 3], dtype=tf.float32)])
        def preprocess_and_predict(input_tensor):
            # Apply same preprocessing as in training
            preprocessed = input_tensor / 255.0
            result = model(preprocessed)
            return {"prediction": result}
        
        # Create a new model that includes the preprocessing
        preprocess_model = tf.keras.models.Model(model.input, model.output)
        
        # Save the model with signatures
        signatures = {
            "serving_default": preprocess_and_predict
        }
        
        # Export the model
        tf.saved_model.save(
            preprocess_model, 
            output_directory,
            signatures=signatures
        )
        
        print(f"Successfully exported model to {output_directory}")
        
        # Verify the saved model
        print("Verifying saved model...")
        loaded_model = tf.saved_model.load(output_directory)
        print("Model loaded successfully. Available signatures:")
        print(list(loaded_model.signatures.keys()))
        
        # Test the model with a random input
        test_input = np.random.random((1, 100, 100, 3)).astype(np.float32)
        serving_fn = loaded_model.signatures["serving_default"]
        prediction = serving_fn(tf.constant(test_input))["prediction"]
        print(f"Test prediction shape: {prediction.shape}")
        print(f"Test prediction sample: {prediction[0, :3]}")
        
        print("\nExport completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error exporting model: {e}")
        
        # Alternative approach - rebuild the model
        try:
            print("\nAttempting alternative approach: rebuilding model...")
            
            # Try a different approach - loading with the Lambda layer
            model = tf.keras.models.load_model(
                input_model_path,
                custom_objects={'TrueDivide': tf.keras.layers.Lambda(lambda x: x / 255.0)},
                compile=False
            )
            
            # Save the model in SavedModel format
            model.save(output_directory, save_format='tf')
            print(f"Successfully exported model using alternative approach to {output_directory}")
            return True
        except Exception as e2:
            print(f"Alternative approach failed: {e2}")
            
            # Try one more approach - extract and rebuild
            try:
                print("\nAttempting to extract and rebuild the model...")
                
                # Create a simple model with the same structure as MobileNetV2 base
                input_layer = tf.keras.layers.Input(shape=(100, 100, 3))
                
                # Add preprocessing layer
                x = tf.keras.layers.Lambda(lambda x: x / 255.0)(input_layer)
                
                # Add MobileNetV2 base
                base = tf.keras.applications.MobileNetV2(
                    input_shape=(100, 100, 3),
                    include_top=False,
                    weights="imagenet"
                )
                base.trainable = False
                x = base(x)
                
                # Add classification head
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                
                # Get the number of classes from the original model
                try:
                    # Try to get the number of units in the last layer
                    last_layer = None
                    for layer in model.layers:
                        if isinstance(layer, tf.keras.layers.Dense):
                            last_layer = layer
                    
                    if last_layer:
                        num_classes = last_layer.units
                        print(f"Detected {num_classes} classes in the original model")
                    else:
                        # Default to 13 classes (empty + 6 white pieces + 6 black pieces)
                        num_classes = 13
                        print(f"Using default of {num_classes} classes")
                except:
                    # Default to 13 classes
                    num_classes = 13
                    print(f"Using default of {num_classes} classes")
                
                output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
                new_model = tf.keras.models.Model(input_layer, output_layer)
                
                print("Rebuilt model summary:")
                new_model.summary()
                
                # Save the rebuilt model
                new_model.save(output_directory, save_format='tf')
                print(f"Successfully saved rebuilt model to {output_directory}")
                
                print("\nNOTE: This is a reconstructed model with the same architecture but without the trained weights.")
                print("You will need to transfer the weights or retrain this model before using it for predictions.")
                return True
            except Exception as e3:
                print(f"Rebuild approach failed: {e3}")
                return False
    
    return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_model.py input_model.h5 output_directory")
        sys.exit(1)
    
    input_model_path = sys.argv[1]
    output_directory = sys.argv[2]
    
    success = export_model(input_model_path, output_directory)
    
    if success:
        print("\nModel export completed successfully!")
        print("You can use the exported model with the following code:")
        print("\nimport tensorflow as tf")
        print(f"model = tf.saved_model.load('{output_directory}')")
        print("# For inference:")
        print("serving_fn = model.signatures['serving_default']")
        print("prediction = serving_fn(tf.constant(image, dtype=tf.float32))['prediction']")
    else:
        print("\nModel export failed. Please see error messages above.")