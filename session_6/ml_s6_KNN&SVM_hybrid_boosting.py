import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

X,y = load_digits(return_X_y=True)

model_1 = SVC(probability=True)
model_1.fit(X,y)
pred_proba = model_1.predict_proba(X)
pred_1 = model_1.predict(X)
print(classification_report(y, pred_1))

model_2 = KNeighborsClassifier(n_neighbors=5)
model_2.fit(pred_proba,y)


pred = model_2.predict(pred_proba)
print(classification_report(y, pred))