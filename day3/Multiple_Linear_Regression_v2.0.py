"""
    Function:Multiple_Linear_Regression
    Author:Will
    Date:2019-1-15
    Version:2.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 4].values

    # imputer = Imputer()
    # imputer = imputer.fit(X[:, 1:4])
    # X[:, 1:4] = imputer.fit_transform(X[:, 1:4])

    labelencoder_X = LabelEncoder()
    X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
    onehotencoder = OneHotEncoder(categorical_features=[3])
    X = onehotencoder.fit_transform(X).toarray()

    # Avoiding Dummy Variable Trap
    X = X[:, 1:]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    Y_pred = regressor.predict(X_test)

    print('Done!')


if __name__ == '__main__':
    main()
