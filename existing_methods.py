
import cv2
import matplotlib.pyplot as plt
from Align import AlignDlib
from landMarks import download_landmarks
from pruning import prune
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, roc_curve, auc

import os
from imutils import paths

import random
import warnings

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = [10, 5]
warnings.filterwarnings("ignore", category=FutureWarning)

dataset = 'lfw'
alignment = AlignDlib('models/landmarks.dat')

# download_landmarks()
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

# defining models

LR = LogisticRegression()
SVM = SVC(probability=True)
RF = RandomForestClassifier(n_estimators=1500)
KNN = KNeighborsClassifier()
MLP = MLPClassifier(hidden_layer_sizes=(1024, 1024,))
AB = AdaBoostClassifier(n_estimators=1000)
GB = GradientBoostingClassifier(n_estimators=1000)
DT = DecisionTreeClassifier(criterion="entropy")


# function for training and prediction

def train_predict(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    acc_test = accuracy_score(testY, y_pred)
    return acc_test


LR_acc = train_predict(LR, trainX, trainY, testX, testY)
print("\n\nThe accuracy score of Logistic Regression is :  %f" % LR_acc)
SVM_acc = train_predict(SVM, trainX, trainY, testX, testY)
print("The accuracy score of SVM is :  %f" % SVM_acc)
RF_acc = train_predict(RF, trainX, trainY, testX, testY)
print("The accuracy score of Random Forest Classifier is :  %f" % RF_acc)
KNN_acc = train_predict(KNN, trainX, trainY, testX, testY)
print("The accuracy score of K-nearest neighbour is :  %f" % KNN_acc)
MLP_ac = train_predict(MLP, trainX, trainY, testX, testY)
print("The accuracy score of Multilayer Perceptron is :  %f" % MLP_ac)
AB_ac = train_predict(AB, trainX, trainY, testX, testY)
print("The accuracy score of adaboosting is:  %f" % AB_ac)
GB_ac = train_predict(GB, trainX, trainY, testX, testY)
print("The accuracy score of gradient boosting is :  %f" % GB_ac)
DT_ac = train_predict(DT, trainX, trainY, testX, testY)
print("The accuracy score of decision tree is :  %f" % DT_ac)
