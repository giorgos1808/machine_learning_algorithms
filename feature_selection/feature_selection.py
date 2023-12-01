# HOMEWORK 10 - Model Evaluation
# Pearson & Spearman correlation
# Tsakiris Giorgos

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np

feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 't']
HTRU_2 = pd.read_csv('HTRU_2.csv', names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 't'])

X = HTRU_2.iloc[:, :-1]
y = HTRU_2.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_predict = dt_model.predict(X_test)

# method 1
importance = dt_model.feature_importances_
imp = []
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    imp.append(v)
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# method 2
correlation=X.corr()
print(correlation)

# task 1
print("Accuracy: %0.4f" % accuracy_score(y_test, y_predict))
print("Recall: %0.4f" % recall_score(y_test, y_predict, average="macro"))
print("Precision: %0.4f" % precision_score(y_test, y_predict, average="macro"))
print("F1: %0.4f" % f1_score(y_test, y_predict, average="macro"))

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# PCA
pca_model = PCA(n_components=4)
X_train_pca = pca_model.fit_transform(X_train)
X_test_pca = pca_model.transform(X_test)

dt_model_pca = DecisionTreeClassifier()
dt_model_pca.fit(X_train_pca, y_train)
y_predict_pca = dt_model_pca.predict(X_test_pca)
print('With PCA')
print("Accuracy: %0.4f" % accuracy_score(y_test, y_predict_pca))
print("Recall: %0.4f" % recall_score(y_test, y_predict_pca, average="macro"))
print("Precision: %0.4f" % precision_score(y_test, y_predict_pca, average="macro"))
print("F1: %0.4f" % f1_score(y_test, y_predict_pca, average="macro"))

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# task 2
importance = dt_model.feature_importances_
imp_np = np.array(imp)
i_sort = np.argsort(-imp_np)
v_sort = -np.sort(-imp_np)
print('Sorted features')
for i in range(len(i_sort)):
    print('Feature: %0d, Score: %.5f' % (i_sort[i], v_sort[i]))

# task 3
X_2 = pd.DataFrame([HTRU_2.iloc[:,i_sort[0]], HTRU_2.iloc[:,i_sort[1]], HTRU_2.iloc[:,i_sort[2]], HTRU_2.iloc[:,i_sort[3]]]).transpose()

X_2_train, X_2_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_2_train, y_train)
y_2_predict = dt_model.predict(X_2_test)

print("Accuracy: %0.4f" % accuracy_score(y_test, y_2_predict))
print("F1-Score: %0.4f" % f1_score(y_test, y_2_predict, average="macro"))
print("Recall: %0.4f" % recall_score(y_test, y_2_predict, average="macro"))
print("Precision: %0.4f" % precision_score(y_test, y_2_predict, average="macro"))

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

