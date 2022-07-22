
import cv2
import matplotlib.pyplot as plt
from Align import AlignDlib
from pruning import prune
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

import os
from imutils import paths

import random
import warnings

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [10, 5]
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = 'lfw'
alignment = AlignDlib('models/landmarks.dat')

prune(dataSet=dataset, threshold=15)

print("Loading images...")
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []

for i, imagePath in enumerate(imagePaths):
    try:
        image = cv2.imread(imagePath, 0)
        image = image[..., ::-1]
        image = alignment.align(96, image, alignment.getLargestFaceBoundingBox(image),
                                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        image = (image / 255.).astype(np.float32)
        data.append(image)
        #labels.append(imagePath.split('/')[-2])
        labels.append(imagePath.split(os.path.sep)[-2])
    except:
        pass

le = LabelEncoder().fit(labels)
Y = le.transform(labels)
Y = np.array(Y)

data = np.array(data)
data = data.reshape(data.shape[0], -1)

print("Input data shape : ", data.shape)
print("Output data shape : ", Y.shape)

# splitting dataset
trainX, testX, trainY, testY = train_test_split(data, Y, test_size=0.3)

print("Input shape : ", trainX.shape)
print("Output shape : ", trainY.shape)

# defining pipeline with PCA and logistic regression
model = Pipeline([('pca', PCA()), ('rf', LogisticRegression())])
model.fit(trainX,trainY)

ypred = model.predict(testX)
accuracy = accuracy_score(testY,ypred)
print("The accuracy score in PCA : ",accuracy)