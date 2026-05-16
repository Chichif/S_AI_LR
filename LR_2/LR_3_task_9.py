import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Завантаження вхідних даних із текстового файлу
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна для Х за допомогою квантиля розподілу відстаней точок
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Кластеризація даних методом зсуву середнього (Mean Shift) із розрахованим вікном
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витягування центрів кластерів (координат центроїдів)
cluster_centers = meanshift_model.cluster_centers_

# Оцінка кількості кластерів на основі унікальних міток, знайдених моделлю
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))

print("Кількість кластерів вхідних даних =", num_clusters)
print("\nЦентри кластерів:\n", cluster_centers)

# Відображення на графіку точок та центрів кластерів
plt.figure()
markers = cycle('o*xvs')

for i, marker in zip(range(num_clusters), markers):
    # Відображення точок, що належать до поточного кластера
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black', s=50)
    
    # Відображення на графіку центру поточного кластера (великий круглий маркер)
    cluster_center = cluster_centers[i]
    plt.scatter(cluster_center[0], cluster_center[1], marker='o', 
                facecolors='black', edgecolors='black', s=300, linewidths=2, zorder=10)

plt.title('Кластеризація методом зсуву середнього (Mean Shift)')
plt.show()