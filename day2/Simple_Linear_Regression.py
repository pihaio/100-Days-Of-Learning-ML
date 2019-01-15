"""
    Function:Simple linear regression
    Author:Will
    Date:2019-1-15
    Version:1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Using pandas read datasets from csv
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, 1].values

# Split datasets into training sets and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

# Fitting simple linear regression model to the training sets
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# Visualization
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, regressor.predict(X_test), color='black')