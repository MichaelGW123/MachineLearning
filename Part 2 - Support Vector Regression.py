# Support Vector Regression

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
y = y.reshape(len(y), 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the Support Vector Regression Model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
y_1d =pd.DataFrame(y).values.ravel()
regressor.fit(X, y_1d)

# Predicting a new result with Support Vector Regression
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))

# Visualizing the Support Vector Regression Results
from sklearn.metrics import r2_score # The more related the two variables, the closer to 1 the r2 score becomes
print(r2_score(sc_y.inverse_transform(y), sc_y.inverse_transform(regressor.predict(X))))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Salary vs Level (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Support Vector Regression Results (Smoother)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X_grid), sc_y.inverse_transform(regressor.predict(X_grid)), color = 'blue')
plt.title('Salary vs Level (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()