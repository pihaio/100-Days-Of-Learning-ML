"""
    Function:pre-process data
    Author:Will
    Date:2019-1-15
    Version:1.0
"""

import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# read data from csv
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 3].values

# Using sklearn.preprocessing.Imputer to handle missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Using sklearn.preprocessing encode categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Create a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Using sklearn.model_selection.train_test_split to split the datasets into training sets and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Using sklearn.preprocessing.StandardScalar to scale feature
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
