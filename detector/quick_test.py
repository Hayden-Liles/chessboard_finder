#!/usr/bin/env python3
"""
Quick test script for the chess piece model.
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
import json

# If you have an older model with a Normalizer layer, we still need to register it.
class Normalizer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / 255.0

    def get_config(self):
        config = super().get_config()
        return config

def load_model(model_path):
    """
    Load a model that may use either:
      - A custom Normalizer layer, or
      - A Lambda(preprocess_input) layer
    by registering both as custom_objects.
    """
    custom_objects = {
        'Normalizer': Normalizer,
        # register the function name exactly as it was serialized:
        'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
    }
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        print(f"✅ Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Test chess piece recognition model"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to your trained model file (e.g. chess_piece_model.keras)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to a single 100×100 test image"
    )
    args = parser.parse_args()

    # 1) Load the model
    model = load_model(args.model)
    if model is None:
        return

    # 2) Load & prep the test image
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Failed to load image at {args.image}")
        return

    # Convert BGR→RGB and resize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (100, 100))
    img_batch = np.expand_dims(img_resized, axis=0)  # shape (1,100,100,3)

    # 3) Predict
    prediction = model.predict(img_batch, verbose=0)[0]

    # 4) Get class names from metadata if available
    metadata_path = args.model + ".metadata"
    class_mapping = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                md = json.load(f)
                class_mapping = md.get("class_mapping", {})
        except Exception as e:
            print(f"⚠️  Could not load metadata: {e}")

    if class_mapping:
        # turn {"0":"b_b","1":"b_w",…} into an ordered list
        class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    else:
        # fallback
        class_names = [
            'b_b','b_w','empty',
            'k_b','k_w','n_b','n_w',
            'p_b','p_w','q_b','q_w',
            'r_b','r_w','unknown'
        ]

    # 5) Print out all probabilities
    print("\nPrediction probabilities:")
    for i, prob in enumerate(prediction):
        name = class_names[i] if i < len(class_names) else f"unknown_{i}"
        print(f"  {name:>5}: {prob:.6f}")

    # 6) Top pick
    idx = int(np.argmax(prediction))
    top_name = class_names[idx] if idx < len(class_names) else "unknown"
    print(f"\n🔍 Predicted class: {top_name} (confidence {prediction[idx]:.4f})")

    # 7) Show your class_mapping dict, if any
    if class_mapping:
        print("\nClass mapping from metadata:")
        for k, v in sorted(class_mapping.items(), key=lambda x: int(x[0])):
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()