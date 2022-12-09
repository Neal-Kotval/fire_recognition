import tensorflow as tf
import numpy as np
import cv2


def prep(img):
    res = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img_array = np.asarray(res)
    img_array = np.expand_dims(img_array, axis=0)
    return res

def predict_fire(img, modelp):
    interpreter = tf.lite.Interpreter(model_path=modelp)

    img_array = prep(img)

    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.image.resize(img_array, [256,256])
    # img_array = tf.expand_dims(img_array, 0)
    print(img_array)

    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    score = tf.nn.softmax(predictions)
    return (-np.argmax(score)+1)
  
vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    print(predict_fire(img=frame, modelp="/Users/nealkotval/fire_recognition/tflite_store/model_15epochs.tflite"))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
