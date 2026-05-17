import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 500

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

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

kernels = ['poly', 'rbf', 'sigmoid']

for kernel in kernels:
    if kernel == 'poly':
        classifier = SVC(kernel=kernel, degree=8, random_state=0)
    else:
        classifier = SVC(kernel=kernel, random_state=0)
        
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print(f"----- Ядро: {kernel.upper()} -----")
    print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
    print(f"F1-Score: {round(f1_score(y_test, y_pred, average='weighted') * 100, 2)}%\n")