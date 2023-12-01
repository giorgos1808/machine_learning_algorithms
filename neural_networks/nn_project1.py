# Project 9 Neural Networks
# Tsakiris Giorgos

from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

breastCancer = load_breast_cancer()
X = breastCancer.data
y = breastCancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

p1 = [10, 'relu', 'sgd', 0.0001, 100]
p2 = [20, 'tanh', 'sgd', 0.0001, 100]
p3 = [20, 'tanh', 'adam', 0.00001, 100]
p4 = [(50, 50, 50), 'relu', 'adam', 0.00001, 100]
p5 = [50, 'tanh', 'lbfgs', 0.00001, 100]
p6 = [(100, 100, 100), 'relu', 'lbfgs', 0.00001, 100]

p = p1

model = MLPC(hidden_layer_sizes=p[0], activation=p[1], solver=p[2], tol=p[3], max_iter=p[4])
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print('hidden_layer_sizes='+str(p[0])+', activation='+str(p[1])+', solver='+str(p[2])+', tol='+str(p[3])+', max_iter='+str(p[4]))

print('Accuracy: %0.4f' % accuracy_score(y_test, y_predicted))
print('F1-Score: %0.4f' % f1_score(y_test, y_predicted))
print('Precision: %0.4f' % precision_score(y_test, y_predicted))
print('Recall: %0.4f' % recall_score(y_test, y_predicted))


