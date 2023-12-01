# HOMEWORK 11 - Ensembles
# Tsakiris Giorgos

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


breastCancer = load_breast_cancer()
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = BaggingClassifier()
est_txt = ['DecisionTree', 'KNN', 'SVC', 'RFC_Gini', 'RFC_Entropy']
estimator = [DecisionTreeClassifier(), KNeighborsClassifier(), SVC()]
n_estimators = [10, 100, 500]

accuracy, precision, recall, f1 = [], [], [], []
for e in estimator:
    for n in n_estimators:
        model.set_params(base_estimator=e, n_estimators=n)
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_predicted))
        precision.append(precision_score(y_test, y_predicted))
        recall.append(recall_score(y_test, y_predicted))
        f1.append(f1_score(y_test, y_predicted))

rfc = RandomForestClassifier()
for c in ['gini', 'entropy']:
    for n in n_estimators:
        rfc.set_params(criterion=c, n_estimators=n)
        rfc.fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_predicted))
        precision.append(precision_score(y_test, y_predicted))
        recall.append(recall_score(y_test, y_predicted))
        f1.append(f1_score(y_test, y_predicted))

array = np.concatenate((np.array(accuracy).reshape((-1,1)), np.array(precision).reshape((-1,1)),
                        np.array(recall).reshape((-1,1)), np.array(f1).reshape((-1,1))), axis=1)

array_df = pd.DataFrame(array, columns=['accuracy', 'precision', 'recall', 'f1'])

estimators_list = []
for e in est_txt:
    for n in n_estimators:
        estimators_list.append(e+'_'+str(n))

array_df['estimators'] = estimators_list
array_df = array_df.set_index('estimators')
print(array_df)

array_df.plot(kind='bar', stacked=False, title='BaggingClassifier ( Algorithm_Estimators )', rot=0)
plt.show()

