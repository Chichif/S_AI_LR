import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    min_x, max_x = X[:, 0].min() - 1, X[:, 0].max() + 1
    min_y, max_y = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, 0.01),
        np.arange(min_y, max_y, 0.01)
    )

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.contourf(x_vals, y_vals, output, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')