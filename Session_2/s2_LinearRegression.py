import yfinance as yf
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

btc = yf.download("BTC-USD")

X = np.arange(len(btc)).reshape(-1, 1)
y = btc["Close"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= False)



# Create Model
model = LinearRegression()

# Train Model
model.fit(x_train, y_train)

# Evaluate Model
y_pred = model.predict(y_test)
error = root_mean_squared_error(y_test, y_pred)
print(error)


#
slope = model.coef_
bias = model.intercept_
print(f"y = {slope} . X + {bias}")

# Visualize
predict = model.predict(X)

plt.plot(X, y, "g", label = "Real Data")
plt.plot(X, predict, "b--", label = "Predicted Data")

plt.legend()
plt.show()
