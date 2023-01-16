#imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imghdr
import os
import numpy as np
from datetime import date

# gets todays date for model release number
today = date.today()

# sets up parameters for neural network
batch_size = 32
img_height = 256 
img_width= 256 

# specifies directory in which to retrieve data
data_path = "./fire_data/data"

# line to prevent out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

#sets up training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split = 0.2, # training: 80% , testing: 20%
  subset = "training",
  seed = 101, # allows us to keep same random set for both validation and testing
  image_size = (img_height, img_width),
  batch_size = batch_size 
)

#sets up testing dataset
validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_path,
  validation_split=0.2, # training: 80% , testing: 20%
  subset="validation",
  seed=101, # allows us to keep same random set for both validation and testing
  image_size=(img_height, img_width),
  batch_size=batch_size
)

#stores class names in order for us to set fire = 0, non_fire = 1
class_names = train_ds.class_names

# preprocessing
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # caches certain operations to make run faster for training dataset
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE) # caches certain operations to make run faster for validation dataset

num_classes = len(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# builds model, utilizing rectified linear as activation
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_classes, name="output")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# sets amount of epochs for dataset 
epochs = 15

# passes datasets into model for training
history = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# predict function
def predict_fire(img_path, model, resize = True, output = False, imshow = False):
    img = tf.keras.utils.load_img(
        img_path
    )
    
    img_array = tf.keras.utils.img_to_array(img)
    
    if (resize):
        img_array = tf.image.resize(img_array, [256,256])
    
    if (imshow):
        img_array_temp = img_array
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array, verbose = 0)
    score = tf.nn.softmax(predictions[0])
    
    if (output):
        print(score)

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    if (imshow):
        plt.imshow(mpimg.imread(img_path))
        
    return class_names[np.argmax(score)]


# calls predict function as a print statement

#saves as tf-lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#stores model release as model(date)_epoch(# of epochs used)
with open("tflite_store/model{}_epoch{}".format(today.strftime("%m-%d-%y"), epochs), 'wb') as f:
  f.write(tflite_model)