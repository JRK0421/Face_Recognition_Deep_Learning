from model import create_model
from Align import AlignDlib
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np
import imutils
import cv2

alignment = AlignDlib('models/landmarks.dat')
model = create_model()
le = LabelEncoder()

model.load_weights('./models/pretrainedModel.h5')
graph = tf.get_default_graph()
CNNmodel = load_model('./models/LFW_CNN.model')
le.classes_ = np.load('./models/labelEncoder.npy')

path = "colin2.jpeg"
# img = cv2.imread(path, 1)
# img = img[..., ::-1]
#

def display(img):
    plt.grid()
    plt.title("Original Query Image")
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    detectFace = alignment.getLargestFaceBoundingBox(img)
    x = detectFace.left()
    y = detectFace.top()
    w = detectFace.width()
    h = detectFace.height()

    fig,ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x,y), x+w, y+h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.set_facecolor('white')
    plt.grid()
    plt.title("Face Detected Image")
    plt.axis('off')
    plt.savefig('faceDetectedOutput.png')
    plt.show()


def prediction(img):
    img = cv2.imread(img, 1)
    img = img[..., ::-1]

    orig = img.copy()
    display(img)
    faces = alignment.getAllFaceBoundingBoxes(img)

    for i in range(len(faces)):
        face_aligned = alignment.align(96, img, faces[i], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        face_aligned = (face_aligned / 255.).astype(np.float32)
        data = np.expand_dims(face_aligned, axis=0)
        global graph
        with graph.as_default():
            embedding = model.predict(data)
            pred = CNNmodel.predict([embedding])
        ind = np.argsort(pred[0])
        name = le.inverse_transform([ind[::-1][0]])[0]
        score = "{0:.2f}%".format(pred[0][ind[::-1][0]]*100)
        output = imutils.resize(orig, width=400)
        out = "{}: {}".format(name,score)
        color = (255, 0, 0)
        cv2.putText(output, out, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./templates/output.jpeg", output)
        # plt.imshow(output)
        # plt.show()
        return name, score


