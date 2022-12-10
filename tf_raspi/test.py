from pathlib import Path
import time


try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter

import numpy as np
import cv2


def predict_fire(img, pret):

    interpreter = pret
    interpreter.allocate_tensors()
    img_res = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    img_array = np.array([img_res], dtype=np.float32)


    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    if predictions[0][0] > predictions [0][1]:
      predictions[0][0] = 1;
      predictions[0][1] = 0
    else:
      predictions[0][1] = 1;
      predictions[0][0] = 0


    #switches inputs 0 <-> 1, 1 = fire, 0 = non-fire
    return (-np.argmax(predictions)+1)
  
vid = cv2.VideoCapture(0)
path = Path("tflite_store/model_15epochs.tflite").absolute()
interpreter = Interpreter(model_path=str(path))

try:
  while(True):
      ret, frame = vid.read()
      if ret == True:
        cv2.imshow('Frame', frame)
        # my_signature = interpreter.get_signature_runner()

        # print(path.absolute())
        store = predict_fire(img=frame, pret=interpreter)
        
        print(store)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
      else:
          break

except KeyboardInterrupt:
  print("Cleaned Up!")
  vid.release()
  cv2.destroyAllWindows()
