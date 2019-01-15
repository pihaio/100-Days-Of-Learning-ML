"""
    Function:Logistic Regression
    Author:Will
    Date:2019-1-15
    Version:1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def main():
    dataset = pd.read_csv("Social_Network_Ads.csv")
    X = dataset.iloc[:, 2:4].values
    Y = dataset.iloc[:, 4].values

    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4, random_state=0)

    classifier = LogisticRegression()
    classifier = classifier.fit(X_train, Y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(Y_test, y_pred)

    X_set, y_set = X_train, Y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Training set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()

    X_set, y_set = X_test, Y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Test set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()

    print('Done!')


if __name__ == '__main__':
    main()
