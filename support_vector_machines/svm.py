# HOMEWORK 6 - Support Vector Machines
# SVM ALGORITHM TEMPLATE
# Tsakiris Giorgos

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

myData = pd.read_csv('creditcard.csv')

X = myData.iloc[:, :-1]
y = myData.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

SVC_model = SVC(C=0.1, kernel='poly', gamma=0.2, degree=2)
SVC_model.fit(X_train, y_train)

y_predict = SVC_model.predict(X_test)

print('Accuracy: %0.4f' % accuracy_score(y_test, y_predict))
print('F1-Score: %0.4f' % f1_score(y_test, y_predict, average='macro'))
print('Precision: %0.4f' % precision_score(y_test, y_predict, average='macro'))
print('Recall: %0.4f' % recall_score(y_test, y_predict, average='macro'))
