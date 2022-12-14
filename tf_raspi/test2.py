from pathlib import Path
import time
from PIL import Image

try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter

import numpy as np
import cv2

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  print(interpreter.get_output_details())
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  
  return [(i, output[i]) for i in ordered[:top_k]][0]



model_path = "/Users/nealkotval/fire_recognition/tflite_store/model_15epochs.tflite"

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

# Load an image to be classified.
# image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))

# Classify the image.
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    if ret == True:
        ret, frame = vid.read()
        image = Image.fromarray(frame).convert('RGB').resize((256, 256))
        time1 = time.time()
        label_id, prob = classify_image(interpreter, image)
        print("PROB: " + str(prob))
        print("LABELID: " + str(label_id))
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classification Time =", classification_time, "seconds.")

        # Read class labels.
        labels = ['fire', 'nonfire']
        # Return the classification label of the image.
        classification_label = labels[label_id]
        print("Image Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
    else:
        break