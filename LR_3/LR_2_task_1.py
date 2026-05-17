import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        
        if '?' in line:
            continue
            
        data = line.strip().split(', ')
        
        if len(data) > 0 and data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif len(data) > 0 and data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X_final = X_encoded[:, :-1].astype(int)
y_final = X_encoded[:, -1].astype(int)
y_label_encoder = label_encoder[-1]

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=5)

classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

print("----- Метрики якості класифікації -----")
print(f"Accuracy (Акуратність): {round(accuracy_score(y_test, y_test_pred) * 100, 2)}% ")
print(f"Precision (Точність): {round(precision_score(y_test, y_test_pred, average='weighted') * 100, 2)}%")
print(f"Recall (Повнота): {round(recall_score(y_test, y_test_pred, average='weighted') * 100, 2)}%")

f1_scores = cross_val_score(classifier, X_final, y_final, scoring='f1_weighted', cv=3)
print(f"F1 score (Крос-валідація): {round(100 * f1_scores.mean(), 2)}%")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '40', '0', '0', 'United-States']

input_data_encoded = np.zeros(len(input_data))
categorical_count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[categorical_count].transform([input_data[i]])[0])
        categorical_count += 1

input_data_encoded = input_data_encoded.reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)
result_label = y_label_encoder.inverse_transform(predicted_class)[0]

print("\n----- Результат для тестової точки -----")
print(f"Прогнозований клас доходу: {result_label}")