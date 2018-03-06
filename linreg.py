# Code partially from a blog:
# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# Modifications was made accordingly here

import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from data import load_data_four_points
from sklearn.model_selection import KFold


if __name__ == '__main__':

    inputs, labels = load_data_four_points()
    kf = KFold(n_splits=3, shuffle=True)

    total_loss = np.zeros((1,))
    # print(total_loss)

    for train, val in kf.split(labels):
        X_train, y_train, X_val, y_val = inputs[train], labels[train], inputs[val], labels[val]
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        score = linreg.score(X_val, y_val)
        print("3-fold cross validation average accuracy: %.3f" % score)
