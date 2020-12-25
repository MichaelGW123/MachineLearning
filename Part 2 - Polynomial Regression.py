# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Importing the dataset
path = Path(__file__).parent / 'Position_Salaries.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression Model on the whole dataset
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X, y)

# Training the Polynomial Regression Model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomialRegressor = PolynomialFeatures(degree = 4)
X_poly = polynomialRegressor.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression Results
from sklearn.metrics import r2_score # The more related the two variables, the closer to 1 the r2 score becomes
print(r2_score(y, linearRegressor.predict(X)))
plt.scatter(X, y, color = 'red')
plt.plot(X, linearRegressor.predict(X), color = 'blue')
plt.title('Salary vs Level (Linear)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression Results
print(r2_score(y, lin_reg_2.predict(X_poly)))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Salary vs Level (Polynomial)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression Results (for higher resolution and smoother curse)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(polynomialRegressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
y_pred = linearRegressor.predict([[6.5]])
print(y_pred)

# Predicting a new result with Polynomial Regression
y_pred = lin_reg_2.predict(polynomialRegressor.fit_transform([[6.5]]))
print(y_pred)