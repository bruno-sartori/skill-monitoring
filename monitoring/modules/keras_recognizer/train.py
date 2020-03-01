import os
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np


def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)
    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)
    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)
    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)
    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == "__main__":
    im = Image.open('A.J._Buckley.jpg')
    im = im.resize((224, 224))
    im = np.array(im).astype(np.float32)
    #    im[:,:,0] -= 129.1863
    #    im[:,:,1] -= 104.7624
    #    im[:,:,2] -= 93.5940
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = vgg_face('vgg-face-keras-fc.h5')
    out = model.predict(im)
    print(out[0][0])


# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Load saved model
classifier_model = tf.keras.models.load_model('face_classifier_model.h5')

dnnFaceDetector = dlib.cnn_face_detection_model_v1(
    "mmod_human_face_detector.dat")

# Label names for class numbers
person_rep = {0: 'Lakshmi Narayana',
              1: 'Vladimir Putin',
              2: 'Angela Merkel',
              3: 'Narendra Modi',
              4: 'Donald Trump',
              5: 'Xi Jinping'}

if __name__ == '__main__':
  file_path = input("Path to image with file size < 100 kb ? ")

  img = cv2.imread(file_path)
  if img is None or img.size is 0:
    print("Please check image path or some error occured")

  else:
    persons_in_img = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Faces
    rects = dnnFaceDetector(gray, 1)
    left, top, right, bottom = 0, 0, 0, 0
    for (i, rect) in enumerate(rects):
      # Extract Each Face
      left = rect.rect.left()  # x1
      top = rect.rect.top()  # y1
      right = rect.rect.right()  # x2
      bottom = rect.rect.bottom()  # y2
      width = right-left
      height = bottom-top
      img_crop = img[top:top+height, left:left+width]
      cv2.imwrite(os.getcwd()+'/crop_img.jpg', img_crop)

      # Get Embeddings
      crop_img = load_img(os.getcwd()+'/crop_img.jpg', target_size=(224, 224))
      crop_img = img_to_array(crop_img)
      crop_img = np.expand_dims(crop_img, axis=0)
      crop_img = preprocess_input(crop_img)
      img_encode = vgg_face(crop_img)

      # Make Predictions
      embed = K.eval(img_encode)
      person = classifier_model.predict(embed)
      name = person_rep[np.argmax(person)]
      os.remove(os.getcwd()+'/crop_img.jpg')
      cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
      img = cv2.putText(img, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
      img = cv2.putText(img, str(np.max(person)), (right, bottom+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
      persons_in_img.append(name)
    # Save images with bounding box,name and accuracy
    cv2.imwrite(os.getcwd()+'/recognized_img.jpg', img)

    #Person in image
    print('Person(s) in image is/are:')
    print(persons_in_img)

    plt.figure(figsize=(8, 4))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()
