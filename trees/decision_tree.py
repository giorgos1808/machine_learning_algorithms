# HOMEWORK 2 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Tsakiris Giorgos

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

breastCancer = load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

Criterion = 'gini'
Max_Depth = 5

model = DecisionTreeClassifier()
model.set_params(criterion=Criterion, max_depth=Max_Depth)
model_fit = model.fit(X_train, y_train)
y_predicted = model.predict(x_test)

print('criterion: %s, max_depth: %d' % (Criterion, Max_Depth))
print('Accuracy: %0.3f' % accuracy_score(y_test, y_predicted))
print('F1-Score: %0.3f' % f1_score(y_test, y_predicted))
print('Precision: %0.3f' % precision_score(y_test, y_predicted))
print('Recall: %0.3f' % recall_score(y_test, y_predicted))

plot_tree(decision_tree=model_fit, feature_names=breastCancer.feature_names[:numberOfFeatures], class_names=breastCancer.target_names, filled=True)
plt.show()
