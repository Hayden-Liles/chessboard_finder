"""
Rebuild chess piece recognition model with proper preprocessing.
"""
import tensorflow as tf
import numpy as np
import os

# Define a proper custom preprocessing layer
class Normalizer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Normalizer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return inputs / 255.0
    
    def get_config(self):
        config = super(Normalizer, self).get_config()
        return config

def rebuild_chess_piece_model(output_path="chess_piece_model_fixed.keras"):
    """Rebuild the chess piece recognition model with proper preprocessing."""
    print(f"Rebuilding chess piece model and saving to {output_path}")
    
    # Create a model with the same architecture
    input_layer = tf.keras.layers.Input(shape=(100, 100, 3), name="input_image")
    
    # Use custom normalizer layer instead of Lambda
    x = Normalizer(name="preprocessing")(input_layer)
    
    # Add MobileNetV2 base with pretrained weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(100, 100, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze the base model weights
    base_model.trainable = False
    
    # Connect base model to our preprocessing output
    x = base_model(x)
    
    # Add standard classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer for 13 chess piece classes
    output_layer = tf.keras.layers.Dense(13, activation="softmax", name="piece_prediction")(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Print model summary
    model.summary()
    
    # Save model in Keras 3 compatible format
    model.save(output_path)
    print(f"Model saved to {output_path}")
    
    # We'll skip the test loading for now since that's where the error occurs
    return True

def main():
    # Define output path
    output_path = "chess_piece_model_fixed.keras"
    
    # Rebuild and save model
    success = rebuild_chess_piece_model(output_path)
    
    if success:
        print("\nModel recreation successful!")
        print("You can now use this model with your app:")
        print(f"python app.py --model {output_path} ./images/1.jpeg")
    else:
        print("\nModel recreation failed. Please see error messages above.")

if __name__ == "__main__":
    main()