from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K
from pyimagesearch.nn.conv import Lenet
import matplotlib.pyplot as plt
import numpy as np
import cv2

((trainX, trainY), (testX, testY)) = mnist.load_data()

if K.image_data_format() == "channels_first":
    trainX = trainX.reshape(trainX.shape[0], 1, 28, 28)
    testX = testX.reshape(testX.shape[0], 1, 28, 28)
else:
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

model = Lenet.build(width=28, height=28, depth=1, classes=10)
opt = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=40, batch_size=128)

preds = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names= [str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.legend()
plt.savefig("lenet_score.png")


