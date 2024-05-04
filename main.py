import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

ticker = yf.Ticker('ES=F')

data = ticker.history(period='1y')
df = pd.DataFrame(data)
df.reset_index(inplace=True)
df = df.drop(columns=['Dividends', 'Stock Splits'])

df['Date'] = pd.to_datetime(df.Date)

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False, random_state=0)

regression = LinearRegression()
regression.fit(X_train, y_train)

regression_confidence = regression.score(X_test, y_test)
print('Linear Regression Confidence: ', regression_confidence)
predicted = regression.predict(X_test)

dfr=pd.DataFrame({'Actual_Price':y_test, 'Predicted_Price':predicted})

# print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, predicted))
# print('Mean Squared Error (MSE) :', metrics.mean_squared_error(y_test, predicted))
# print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, predicted)))

x2 = dfr.Actual_Price.mean()
y2 = dfr.Predicted_Price.mean()
Accuracy1 = x2/y2*100
print("The accuracy of the model is " , Accuracy1)


plt.plot(dfr.Actual_Price, color='black')
plt.plot(dfr.Predicted_Price, color='lightblue')
plt.title("ES prediction chart")
plt.show()