import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")

X = df.drop(columns= "Outcome")
y = df["Outcome"]

print(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify= y)

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

for solver in solvers:
    model = LogisticRegression(max_iter=5000, solver = solver)

    model.fit(x_train, y_train)

    predict = model.predict(X)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    print(solver, accuracy_score(y_test, test_pred))