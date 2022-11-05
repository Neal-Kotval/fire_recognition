import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

def predict_fire(image_path, model_path, class_names = ['fire', 'non_fire'], output = False, resize = True, index = False):
    img = tf.keras.utils.load_img(
        image_path
    )

    img_array = tf.keras.utils.img_to_array(img)
    if (resize):
        img_array = tf.image.resize(img_array, [256,256])
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    interpreter = tf.lite.Interpreter(model_path=model_path)
    classify = interpreter.get_signature_runner('serving_default')
    predictions = classify(sequential_input=img_array)['output']
    score = tf.nn.softmax(predictions)
    if (output):
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

    return (class_names[np.argmax(score)])


imgp = '/Users/nealkotval/fire_recognition/test_cases/nonfire3.jpg'
modelp = '/Users/nealkotval/fire_recognition/src/model_15epochs.tflite'
print(predict_fire(imgp, modelp))

