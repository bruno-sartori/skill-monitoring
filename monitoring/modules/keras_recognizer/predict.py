import cv2
import tensorflow as tf
import numpy as np
import keras

CATEGORIES = ["bruno", "karina", "sabrina", "wilian"]

def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")
image = "bruno1.jpeg"  # your image path


img = np.array(keras.preprocessing.image.load_img(image, target_size=(600, 400)))
prediction = model.predict([img])

#prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])
