import tensorflow as tf
import numpy as np
import cv2


def predict_fire(img, modelp, output = False, resize = True, index = False):
    # modelp = '/Users/nealkotval/fire_recognition/src/model_15epochs.tflite'
    interpreter = tf.lite.Interpreter(model_path=modelp)
    # img = tf.keras.utils.load_img(
    #         img
    # )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.image.resize(img_array, [256,256])
    img_array = tf.expand_dims(img_array, 0)
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    score = tf.nn.softmax(predictions)
    return (-np.argmax(score)+1)
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
  
    cv2.imshow('frame', frame)
    print(predict_fire(img=frame, modelp="/Users/nealkotval/fire_recognition/tflite_store/model_15epochs.tflite"))
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

    # print(predict_fire(frame, interpreter))
