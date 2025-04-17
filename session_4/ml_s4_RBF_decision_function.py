import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

X, y = make_classification(
    n_samples=1000,
    n_features=3,
    n_informative=3,
    n_redundant=0
)

svm_model = SVC(C=5, kernel="rbf")
svm_model.fit(X,y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:,0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired)


xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() -1, X[:, 0].max() + 1, 50),
    np.linspace(X[:, 1].min() - 1, X[:, 0].max() + 1, 50)
)

zz = np.zeros_like(xx)

for i in range(len(xx)):
    for j in range(len(yy)):
        zz[i,j] = svm_model.decision_function([[xx[i,j] ,yy[i,j], 0]])


ax.plot_surface(xx, yy, zz,color='b', alpha=0.8)

ax.set_xlabel("X Axes")
ax.set_ylabel("Y Axes")
ax.set_zlabel("Z Axes")

plt.show()