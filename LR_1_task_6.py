import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Зчитування даних
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

X = data[:, :-1]
y = data[:, -1]

# SVM
svm_model = svm.SVC()
svm_model.fit(X, y)
y_pred_svm = svm_model.predict(X)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X, y)
y_pred_nb = nb_model.predict(X)

# SVM метрики
print("SVM:")
print("Accuracy:", accuracy_score(y, y_pred_svm))
print("Recall:", recall_score(y, y_pred_svm, average='macro'))
print("Precision:", precision_score(y, y_pred_svm, average='macro'))
print("F1:", f1_score(y, y_pred_svm, average='macro'))

# NB метрики
print("\nNaive Bayes:")
print("Accuracy:", accuracy_score(y, y_pred_nb))
print("Recall:", recall_score(y, y_pred_nb, average='macro'))
print("Precision:", precision_score(y, y_pred_nb, average='macro'))
print("F1:", f1_score(y, y_pred_nb, average='macro'))