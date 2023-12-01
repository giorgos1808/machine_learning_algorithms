# HOMEWORK 4 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Tsakiris Giorgos

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

pd.options.mode.chained_assignment = None

random.seed = 42
np.random.seed(42)

titanic = pd.read_csv('titanic.csv')
titanic_drop = titanic.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1)
titanic_drop = titanic_drop.dropna()
titanic_drop = titanic_drop.reset_index()
titanic_drop = titanic_drop.drop('index', axis=1)

X = titanic_drop.drop('Survived', axis=1)
y = titanic_drop['Survived']

for i in range(X.shape[0]):
    if X['Sex'].loc[i] == 'male':
        X['Sex'].loc[i] = 0
    elif X['Sex'].loc[i] == 'female':
        X['Sex'].loc[i] = 1

    if X['Embarked'].loc[i] == 'S':
        X['Embarked'].loc[i] = 0
    elif X['Embarked'].loc[i] == 'C':
        X['Embarked'].loc[i] = 1
    elif X['Embarked'].loc[i] == 'Q':
        X['Embarked'].loc[i] = 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

max = 200
Weights = 'uniform'
Metric = 'minkowski'
Power_parameter = 2
knn_model = KNeighborsClassifier()

accuracy = np.zeros(max-1)
precision = np.zeros(max-1)
recall = np.zeros(max-1)
f1 = np.zeros(max-1)
for n in range(0,max-1):
    knn_model.set_params(n_neighbors=n+1, weights=Weights, metric=Metric, p=Power_parameter)
    model_fit = knn_model.fit(X_train, y_train)
    y_predicted = knn_model.predict(X_test)

    accuracy[n] = accuracy_score(y_test, y_predicted)
    precision[n] = precision_score(y_test, y_predicted)
    recall[n] = recall_score(y_test, y_predicted)
    f1[n] = f1_score(y_test, y_predicted)

best_neighbor = f1.argmax()
print('Weights: %s, Metric: %s, Power_parameter: %s' % (Weights, Metric, Power_parameter))
print('Neighbors count of best F1: %d' % (best_neighbor+1))
print('Accuracy: %f' % accuracy[best_neighbor])
print('F1-Score: %f' % f1[best_neighbor])
print('Precision: %f' % precision[best_neighbor])
print('Recall: %f' % recall[best_neighbor])

plt.title('k-Nearest Neighbors (Weights = '+Weights+', Metric = '+Metric+', p = '+str(Power_parameter)+')')
plt.plot(np.array(range(1, max)), f1, label='f1')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('F1')
plt.show()
