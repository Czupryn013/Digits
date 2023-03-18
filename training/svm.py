from sklearn import datasets, svm, metrics
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train, y_train), (x_test, y_test) = mnist.load_data()
clf = svm.SVC()

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
x = x.reshape(70000, -1)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, shuffle=False
)
#split data into test and train sets
#x = images, y = number values

clf.fit(X_train, y_train) #training
predicted = clf.predict(X_test) #using the model

pickle.dump(clf, open("svm.sav", 'wb'))

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for ax, image, prediction, label in zip(axes, X_test, predicted, y_test):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction} \n True: {label}")
#showing results
plt.show()

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

con_matrix = confusion_matrix(y_test, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix)
disp.plot()

# jupyter-kernelspec uninstall .venv
# ipython kernel install --name=.venv