import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

df = pd.read_csv("diabetes.csv")

X = df.drop(columns = "Outcome")
y = df["Outcome"]

print(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression(max_iter=5000, warm_start=True)

model.fit(x_train, y_train)

y_test_pred_proba = model.predict_proba(x_test)[:,1]

fpr, tpr, threshhold = roc_curve(y_test, y_test_pred_proba)


plt.plot(fpr, tpr, label= "Logistic Regression")
plt.plot([0,1], [0,1], "k--")

plt.xlim([0,1])
plt.ylim([0,1])

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Posotive Rate (TPR)")

plt.title("Receiver operating characteristic (ROC Curve)")

plt.show()