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
image = "pessoa1.jpeg"  # your image path


IMG_SIZE = 50

image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

img = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
img = img/255.0

prediction = model.predict([img])

print('results================')
print(prediction[0])
print(len(prediction[0]))
print(list(prediction[0]))
print(CATEGORIES)

#prediction = model.predict([image])
prediction = list(prediction[0])
print(CATEGORIES[prediction.index(max(prediction))])
