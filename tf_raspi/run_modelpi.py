from pathlib import Path

try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf
  Interpreter = tf.lite.Interpreter

import numpy as np
import cv2


def prep(img):
    img_array = np.asarray(img)
    img_array = cv2.resize(img_array)

    img_array = np.expand_dims(img_array, axis=1)
    return img_array

def predict_fire(img, modelp):

    interpreter = Interpreter(model_path=modelp)
    img_array = np.asarray(cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)


    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    score = tf.nn.softmax(predictions)

    #switches inputs 0 <-> 1, 1 = fire, 0 = non-fire
    return (-np.argmax(score)+1)
  
vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    path = Path("tflite_store/model_15epochs.tflite").absolute()
    # print(path.absolute())
    print(predict_fire(img=frame, modelp=str(path)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
