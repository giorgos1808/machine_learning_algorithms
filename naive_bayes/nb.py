# HOMEWORK 5 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Tsakiris Giorgos

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

textData = fetch_20newsgroups()

X = textData.data
y = textData.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

alpha = 0.25
model = make_pipeline(TfidfVectorizer(norm='l1'), MultinomialNB(alpha=alpha))
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted, average='macro')
precision = precision_score(y_test, y_predicted, average='macro')
f1 = f1_score(y_test, y_predicted, average='macro')
print('Accuracy: %f' % accuracy)
print('F1-Score: %f' % f1)
print('Precision: %f' % precision)
print('Recall: %f' % recall)

confusionMatrix = confusion_matrix(y_test, y_predicted)
sns.heatmap(confusionMatrix.T, square=False, annot=True, fmt='d', cbar=False,
            xticklabels=textData.target_names, yticklabels=textData.target_names)

plt.title('Multinomial NB - Confusion matrix (a = %0.2f)[Acc = %f, Prec = %f, Rec = %f, F1 = %f' % (alpha, accuracy, precision, recall, f1))
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.show()
