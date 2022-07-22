
import cv2
import dlib
import matplotlib.pyplot as plt
from Align import AlignDlib
from landMarks import download_landmarks
from pruning import prune
from model import create_model
import numpy as np
import itertools

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
from sklearn.metrics import confusion_matrix
from keras.initializers import he_normal

import os
import time
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
import warnings

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [10, 5]
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = 'lfw'
alignment = AlignDlib('models/landmarks.dat')

batch_size = 128
nb_epoch = 500

# download_landmarks()
prune(dataSet=dataset, threshold=15)

model = create_model()
model.load_weights('./models/pretrainedModel.h5')
model.summary()
print("Loading images...")

imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

embedded = np.zeros((len(imagePaths), 128))
labels = []

for i, imagePath in enumerate(imagePaths):
    try:
        image = cv2.imread(imagePath, 1)
        image = image[..., ::-1]
        image = alignment.align(96, image, alignment.getLargestFaceBoundingBox(image),
                                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        image = (image / 255.).astype(np.float32)
        embedded[i] = model.predict(np.expand_dims(image, axis=0))[0]
    except:
        print(imagePath)
        pass

    #labels.append(imagePath.split('/')[-2])
    labels.append(imagePath.split(os.path.sep)[-2])

labels = np.array(labels)
print(labels.shape)

le = LabelEncoder().fit(labels)
Y = le.transform(labels)

# save
np.save('./models/labelEncoder.npy', le.classes_)
y = np.zeros((embedded.shape[0], len(set(Y))), dtype=int)

for i, name in enumerate(labels):
    idx = Y[i]
    y[i][idx] = 1

print("Input data shape : ", embedded.shape)
print("Output data shape : ", y.shape)

np.save("./models/inputData.npy", embedded)
np.save("./models/outputData.npy", y)

# splitting dataset
trainX, testX, trainY, testY = train_test_split(embedded, y, test_size=0.3)

print(trainX.shape)
print(trainY.shape)

output_dim = trainY.shape[1]
input_dim = trainX.shape[1]

# training the model with data
model = Sequential()
model.add(Dense(700, input_shape=(input_dim,)))
model.add(Activation('relu'))
model.add(Dense(450))
model.add(Activation('relu'))
model.add(Dense(output_dim))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training
trainedModel = model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epoch, validation_data=[testX, testY])

# plotting the results
plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(trainedModel.history['acc'])
plt.plot(trainedModel.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.subplot(122)
plt.plot(trainedModel.history['loss'])
plt.plot(trainedModel.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# Evaluation
score = model.evaluate(testX, testY, batch_size=batch_size)
print()
print("The accuracy is :- %0.2f" % score[1])

# saving model to JSON
model_json = model.to_json()
with open("./models/LFW_CNN.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./models/LFW_CNN.h5")

# Saving the model
print("Saving network...")
model.save("./models/LFW_CNN.model")

print("Saved model to disk")


'''

predictions = model.predict(testX)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

matrix = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(matrix, classes=set(labels), title='Confusion matrix, without normalization')
plt.show()

'''
