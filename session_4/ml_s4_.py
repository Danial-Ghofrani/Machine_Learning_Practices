from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import time


X,y = make_blobs(n_samples=10000, n_features=3, centers=4, cluster_std=6)

svm_report = []
print("SVM")
start_time = time.time()
svm_clf = SVC()
svm_clf.fit(X,y)
end_time = time.time()
train_time = end_time - start_time
svm_report.append(train_time)

print("Train Time : ",train_time)
start_time = time.time()
pred = svm_clf.predict(X)
end_time = time.time()
predict_time = end_time - start_time
print("Predict Time : ",predict_time)
print(classification_report(y, pred))

print("-------------------------------------------------------")


print("Logistic regression")
start_time = time.time()
reg_clf = LogisticRegression()
reg_clf.fit(X,y)
end_time = time.time()
train_time = end_time - start_time
print("Train Time : ",train_time)
start_time = time.time()
pred = reg_clf.predict(X)
end_time = time.time()
predict_time = end_time - start_time
print("Predict Time : ",predict_time)
print(classification_report(y, pred))
