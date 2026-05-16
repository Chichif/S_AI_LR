import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Завантаження вбудованого набору даних Iris
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Створення об'єкта KMeans з оптимальними параметрами (ініціалізація k-means++, 3 кластери, 10 запусків)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=42)

# Навчання модели кластеризації KMeans на основі матриці ознак X
kmeans.fit(X)

# Передбачення міток кластерів для всіх вхідних точок даних
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації за першими двома ознаками (довжина та ширина чашолистка)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Кластеризація KMeans на наборі Iris (Sklearn)')
plt.show()

# Визначення власної функції для реалізації алгоритму K-Means з нуля
def find_clusters(X, n_clusters, rseed=2):
    # Ініціалізація генератора випадкових чисел та випадковий вибір початкових центрів із точок даних
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # Обчислення відстаней та віднесення кожної точки до найближчого центру за допомогою pairwise_distances_argmin
        labels = pairwise_distances_argmin(X, centers)
        
        # Перерахунок нових центрів як середнього арифметичного координат точок, що увійшли до кожного кластера
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        
        # Перевірка умови збіжності: якщо центри більше не змінюють свої координати, алгоритм завершує роботу
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# Запуск власної функції кластеризації з початковим seed=2 та виведення результату на графік
centers, labels = find_clusters(X, 3, rseed=2)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Власний K-Means (генератор випадковості = 2)')
plt.show()

# Повторний запуск власної функції з іншим seed=0 для демонстрації впливу початкових центрів на фінальний результат
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Власний K-Means (генератор випадковості = 0)')
plt.show()

# Використання вбудованого методу fit_predict з KMeans бібліотеки sklearn для швидкого навчання та отримання міток
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Швидка кластеризація Sklearn (fit_predict)')
plt.show()