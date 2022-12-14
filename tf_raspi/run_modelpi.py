from pathlib import Path
import time
import RPi.GPIO as GPIO
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

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)


def prep(img):
    img_array = np.asarray(img)
    img_array = cv2.resize(img_array)

    img_array = np.expand_dims(img_array, axis=1)
    return img_array

def predict_fire(img, pret):

    interpreter = pret
    interpreter.allocate_tensors()
    prep = Image.fromarray(img).convert('RGB').resize((256, 256))
    # img_array = np.asarray(cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA), dtype=np.float32)
    img_array = np.expand_dims(np.array(np.asarray(prep), dtype=np.float32), axis=0)


    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    # score = tf.nn.softmax(predictions)
    if predictions[0][0] > predictions [0][1]:
      predictions[0][0] = 1;
      predictions[0][1] = 0
    else:
      predictions[0][1] = 1;
      predictions[0][0] = 0

    #switches inputs 0 <-> 1, 1 = fire, 0 = non-fire
    return (-np.argmax(predictions)+1)
  
vid = cv2.VideoCapture(0)
interpreter = Interpreter(model_path=str(path))

try:
  while(True):
      ret, frame = vid.read()
      path = Path("tflite_store/model_15epochs.tflite").absolute()
      # print(path.absolute())
      store = predict_fire(img=frame, pret=interpreter)
      print(store)
      if (store == 1):
        GPIO.output(18, GPIO.HIGH)
      else:
        GPIO.output(18, GPIO.LOW)
      time.sleep(3)

except KeyboardInterrupt:
  print("Cleaned Up!")
  GPIO.cleanup()
  vid.release()
