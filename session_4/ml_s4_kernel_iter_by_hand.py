import time

from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=5000, n_features=3, centers=3, cluster_std=4)

kernels = {'linear', 'poly', 'rbf', 'sigmoid'}
C = {0.01, 0.1, 1, 10, 100}

for kernel in kernels:
    for c in C:
        s_t = time.time()
        clf = SVC(kernel=kernel, C=c)
        clf.fit(X,y)
        print(f"{kernel:8}, {c:5}, {clf.score(X, y):0.2f}, {time.time() - s_t:4f}")


