import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Load trained model
model = tf.keras.models.load_model("accident_model.keras")

IMG_SIZE = 224

def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Image not found.")
        return

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    print("\nPrediction Probability:", round(float(prediction), 4))

    if prediction < 0.5:
        print("Accident Detected")
    else:
        print("Normal Traffic")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py image_path")
    else:
        predict_image(sys.argv[1])
