import numpy as np
from sklearn import preprocessing

# Мітки
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Створення кодувальника
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Відображення
print("Label mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Кодування
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)

print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Декодування
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)

print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))