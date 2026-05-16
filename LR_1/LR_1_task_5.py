import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def find_TP(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 0))

def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def larikov_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP],
                     [FN, TP]])

def larikov_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def larikov_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def larikov_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def larikov_f1_score(y_true, y_pred):
    recall = larikov_recall_score(y_true, y_pred)
    precision = larikov_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

y_true = np.array([1,0,1,1,0,1,0,0,1,0])
y_pred = np.array([1,0,1,0,0,1,1,0,1,0])

print("TP:", find_TP(y_true, y_pred))
print("FN:", find_FN(y_true, y_pred))
print("FP:", find_FP(y_true, y_pred))
print("TN:", find_TN(y_true, y_pred))

print(larikov_confusion_matrix(y_true, y_pred))

print(larikov_accuracy_score(y_true, y_pred))
print(larikov_recall_score(y_true, y_pred))
print(larikov_precision_score(y_true, y_pred))
print(larikov_f1_score(y_true, y_pred))

assert np.array_equal(
    larikov_confusion_matrix(y_true, y_pred),
    confusion_matrix(y_true, y_pred)
)

assert larikov_accuracy_score(y_true, y_pred) == accuracy_score(y_true, y_pred)
assert larikov_recall_score(y_true, y_pred) == recall_score(y_true, y_pred)
assert larikov_precision_score(y_true, y_pred) == precision_score(y_true, y_pred)
assert np.isclose(larikov_f1_score(y_true, y_pred), f1_score(y_true, y_pred))

print("OK")