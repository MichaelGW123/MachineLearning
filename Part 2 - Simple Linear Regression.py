# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Importing the dataset
path = Path(__file__).parent / 'Salary_Data.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Simple Linear Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)

#Predicting the Test Set results
y_pred = regressor.predict(X_test)

#Visualizing the Training Set results
from sklearn.metrics import r2_score # The more related the two variables, the closer to 1 the r2 score becomes
print(r2_score(y_train, regressor.predict(X_train)))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test Set results
print(r2_score(y_test, regressor.predict(X_test)))
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()