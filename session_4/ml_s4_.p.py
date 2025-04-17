from sklearn.svm import SVC
from sklearn.datasets import make_blobs


X,y = make_blobs(n_samples=100, n_features=3, centers=4, random_state=32)

clf = SVC(c=5, kernel="rbf", random_state=32)
clf.fit(X,y)

w = clf.coef_
b = clf.intercept_
