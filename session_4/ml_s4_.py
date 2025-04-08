from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time


X,y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5)

start_time = time.time()
svm_clf = SVC()
svm_clf.fit(X,y)
end_time = time.time()
train_time = end_time - start_time
reg_clf = LogisticRegression()

svm_clf.fit(X,y)
clf.fit(X,y)
