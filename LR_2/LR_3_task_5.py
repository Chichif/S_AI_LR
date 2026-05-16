import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.4 * X ** 2 + X + 4 + np.random.randn(m, 1)

lin_reg_pure = LinearRegression()
lin_reg_pure.fit(X, y)
y_pred_linear = lin_reg_pure.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

print("Intercept:", lin_reg_poly.intercept_)
print("Coefficients:", lin_reg_poly.coef_)

X_new = np.linspace(-3, 3, m).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_new_poly = lin_reg_poly.predict(X_new_poly)
y_new_linear = lin_reg_pure.predict(X_new)

plt.scatter(X, y, color='green', label='Дані')
plt.plot(X_new, y_new_linear, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X_new, y_new_poly, color='black', linewidth=3, label='Поліноміальна регресія')
plt.legend()
plt.show()

y_poly_pred = lin_reg_poly.predict(X_poly)

print("\nLinear Model Metrics:")
print("MAE =", round(sm.mean_absolute_error(y, y_pred_linear), 2))
print("MSE =", round(sm.mean_squared_error(y, y_pred_linear), 2))
print("R2 score =", round(sm.r2_score(y, y_pred_linear), 2))

print("\nPolynomial Model Metrics:")
print("MAE =", round(sm.mean_absolute_error(y, y_poly_pred), 2))
print("MSE =", round(sm.mean_squared_error(y, y_poly_pred), 2))
print("R2 score =", round(sm.r2_score(y, y_poly_pred), 2))