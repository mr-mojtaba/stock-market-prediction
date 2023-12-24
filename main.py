# Need to install ( pip install numpy )
import numpy as np

# Need to install ( pip install pandas )
import pandas as pd

# Need to install ( pip install scikit-learn )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Creating fixed randoms.
np.random.seed(42)

# number of days.
num_data_points = 100

# Creating days.
days = np.arange(1, num_data_points + 1)
print(days)
