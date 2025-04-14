import time

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=5000, n_features=3, centers=3, cluster_std=4, random_state=32)

params = {"kernel" : ['linear', 'poly', 'rbf', 'sigmoid'],
"C" : [0.01, 0.1, 1, 10, 100]}


clf = SVC(verbose=1)
gs_model = GridSearchCV(estimator=clf, param_grid=params, cv=5, verbose= 2)
gs_model.fit(X,y)

print(gs_model.best_params_)
best_model = gs_model.best_estimator_