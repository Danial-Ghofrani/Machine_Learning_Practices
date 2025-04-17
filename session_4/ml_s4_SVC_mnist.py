from keras.datasets import mnist
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

x_train = np.array([np.ravel(x.astype(np.float32)) / 255.0 for x in x_train])
x_test = np.array([np.ravel(x.astype(np.float32)) / 255.0 for x in x_test])

clf = SVC()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

