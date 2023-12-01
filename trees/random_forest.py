# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# Tsakiris Giorgos

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

breastCancer = load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = RandomForestClassifier()

Criterion = 'entropy'
N_Estimators = 100

model.set_params(criterion=Criterion, n_estimators=N_Estimators)
model_fit = model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

print('criterion: %s, n_estimators: %d' % (Criterion, N_Estimators))
print('Accuracy: %0.3f' % accuracy_score(y_test, y_predicted))
print('Precision: %0.3f' % precision_score(y_test, y_predicted))
print('Recall: %0.3f' % recall_score(y_test, y_predicted))
print('F1: %0.3f' % f1_score(y_test, y_predicted))

max = 200
accuracy = np.zeros(max)
precision = np.zeros(max)
recall = np.zeros(max)
f1 = np.zeros(max)

for n in range(0,max):
    model.set_params(criterion='entropy', n_estimators=(n+1) )
    model_fit = model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)

    accuracy[n] = accuracy_score(y_test, y_predicted)
    precision[n] = precision_score(y_test, y_predicted)
    recall[n] = recall_score(y_test, y_predicted)
    f1[n] = f1_score(y_test, y_predicted)

fig = plt.figure('Random Forest Classifier')
plt1, plt2, plt3, plt4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)
plt1.plot(range(1, max+1), accuracy)
plt1.set_xlabel('Number of trees')
plt1.set_ylabel('Accuracy')
plt2.plot(range(1, max+1), precision)
plt2.set_xlabel('Number of trees')
plt2.set_ylabel('Precision')
plt3.plot(range(1, max+1), recall)
plt3.set_xlabel('Number of trees')
plt3.set_ylabel('Recall')
plt4.plot(range(1, max+1), f1)
plt4.set_xlabel('Number of trees')
plt4.set_ylabel('F1-Score')
plt.show()
