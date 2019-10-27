
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import json
import keras
from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers
from keras import models
from keras.models import model_from_json

EPOCHS = 30

def onehot_encoding(labels):
    """ one hot encoding for labels"""
    #dataframe = self.cf
    unique  = np.unique(labels).tolist()
    one_hot_labels = np.zeros((len(labels), len(unique)))
    for i in range(len(labels)):
        #print(cf.iloc[i,loc])
        pitch = labels[i]
        ind = unique.index(pitch)
        one_hot_labels[i, ind] = 1
    return one_hot_labels, unique

labels_dic = json.load(open("labels2641_improved.json", "r"))
print("number files in labels:", len(labels_dic.keys()))

img_arr = []
labels = []
for file in labels_dic.keys():
    # print(file[6:])
    if os.path.exists(file[6:]):
        # print(file[6:])
        img = cv2.imread(file[6:])
        assert(img.shape==(256,256,3))
        img_arr.append(img)
        labels.append(labels_dic[file])

wasser_dir = ""
for file in os.listdir(wasser_dir):
    img = cv2.imread(os.path.join(wasser_dir, file))
    assert(img.shape==(256,256,3))
    img_arr.append(img)
    labels.append("water")

img_arr = np.asarray(img_arr)
print(img_arr.shape)

assert(len(labels)==len(img_arr))
print("unique labels:", np.unique(labels))
num_classes = len(np.unique(labels))

# final data:
img_data = preprocess_input(img_arr.astype(np.float))
labels_one_hot, unique = onehot_encoding(labels)
mapping_dic = {i:unique[i] for i in range(len(unique))}
json.dump(mapping_dic, open("label_mapping.json", "w"))
print(labels_one_hot.shape)

sizes = img_data.shape
print(img_data.shape)

X_train, X_test, y_train, y_test = train_test_split(img_data, labels_one_hot.astype(int), test_size=0.1)

# Vgg + dense pred_model
vggmodel = VGG16(weights='imagenet', include_top=False)
model = models.Sequential()
model.add(layers.InputLayer(input_shape=img_data.shape[1:]))
model.add(vggmodel)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dense(num_classes, activation='softmax'))
# print("trainables", model.trainable_weights)
vggmodel.trainable=False
# print("trainables", model.trainable_weights)
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=keras.losses.categorical_crossentropy,
              metrics=["accuracy"])
model.summary()

model.fit(x = X_train, y= y_train,epochs=EPOCHS, validation_split = 0.2, shuffle=True)


## SAVE MODEL
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
