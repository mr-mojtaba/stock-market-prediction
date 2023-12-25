# Need to install ( pip install numpy )
import numpy as np

# Need to install ( pip install pandas )
import pandas as pd

# Need to install ( pip install scikit-learn )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Need to install ( pip install matplotlib )
import matplotlib.pyplot as plt

# To create fixed randoms.
np.random.seed(42)

# number of days.
num_data_points = 100

# To create days.
days = np.arange(1, num_data_points + 1)

r = np.random.randn(num_data_points)

# Simulation of market fluctuations.
prices = 100 + np.cumsum(r)

# To create a data frame for rows and columns.
data = pd.DataFrame({'Days': days, 'Prices': prices})

# Convert values to float.
data['Prices'] = data['Prices'].astype(float)

# The capital X is the meaning of the attributes.
X = data[['Days']]
# The small y is the meaning of the target variables.
y = data['Prices']

# To divide the data into a training set and a test set.
# test_size -> Division ratio for test data
# random_state -> To create fixed randoms
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# To create an object of MinMaxScaler.
scaler = MinMaxScaler()

# For scaling.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Learning algorithm and specifying its kernel.
model = SVR(kernel='linear')

# For learning to take place.
model.fit(X_train_scaled, y_train)

# Prediction based on the model.
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Comparison of forecasts and actual values.
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# Definition of future days.
new_days = np.arange(num_data_points + 1, num_data_points + 11)

# To create a data frame for rows and columns of new days.
new_days_df = pd.DataFrame({'Days': new_days})

# For new days scaling.
new_days_scaled = scaler.transform(new_days_df)

# Learning algorithm and specifying its kernel.
new_prices = model.predict(new_days_scaled)

# For illustration
# figsize -> For Canvas dimensions
plt.figure(figsize=(12, 5))

# Draw a scatter plot for learning data.
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Draw a scatter plot for testing data.
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Draw a scatter plot for predicting data.
plt.scatter(new_days, new_prices, color='#64B5F6', label='Predicting Data')

# To draw a line graph of training predictions.
plt.plot(X_train, y_pred_train, color='#F4511E', label='Training Prediction')

# To draw a line graph of testing predictions.
plt.plot(X_test, y_pred_test, color='#4E342E', label='Testing Prediction')

plt.xlabel('Days')
plt.ylabel('Prices')
plt.title('Stock Price prediction')

# For a colors guide
plt.legend()

# To display canvas output.
plt.show()
