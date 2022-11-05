import tensorflow as tf
import numpy as np

def predict_fire(img, interpreter, output = False, resize = True, index = False):
    # interpreter = tf.lite.Interpreter(model_path=model_path)
    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    score = tf.nn.softmax(predictions)
    return (np.argmax(score))

imgp = '/Users/nealkotval/fire_recognition/test_cases/nonfire3.jpg'
modelp = '/Users/nealkotval/fire_recognition/src/model_15epochs.tflite'
interpreter = tf.lite.Interpreter(model_path=modelp)
img = tf.keras.utils.load_img(
        imgp
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.image.resize(img_array, [256,256])
img_array = tf.expand_dims(img_array, 0)

print(predict_fire(imgp, interpreter))
