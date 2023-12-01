# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# Tsakiris Giorgos

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

diabetes = load_diabetes()

feature = 2

X = diabetes.data[:, np.newaxis, feature]
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

linearRegressionModel = LinearRegression()
linearRegressionModel.fit(X_train, y_train)
y_predicted = linearRegressionModel.predict(X_test)

pearsonr_coefficient, pearsonr_p_value = pearsonr(y_test, y_predicted)
spearmanr_coefficient, spearmanr_p_value = spearmanr(y_test, y_predicted)

print('Feature: ' + diabetes['feature_names'][feature])
print('PearsonR Coefficient: %0.3f' % pearsonr_coefficient)
print('SpearmanR Coefficient: %0.3f' % spearmanr_coefficient)
print('MSE: %2f' % mean_squared_error(y_test, y_predicted))
print('r^2: %2f' % r2_score(y_test, y_predicted))

plt.plot(X_test, y_predicted, label='Linear Regression', color='b')
plt.scatter(X_test, y_test, label='Actual Test Data', color='r')
plt.title(diabetes['feature_names'][feature] + ' / target')
plt.legend()
plt.show()

