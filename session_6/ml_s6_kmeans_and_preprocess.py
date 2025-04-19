import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)

model = KMeans(n_clusters=10)
model.fit(X)
pred = model.predict(X)

print(y[0])
print(pred[0])

