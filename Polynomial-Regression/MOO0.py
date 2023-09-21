# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # independant values { 1:2 we did it to convert vector to matrix }
y = dataset.iloc[:, 2].values # dependant values


# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Fitting linear Reg to dataSet
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial Reg to dataSet
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
poly_x = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_x, y)

#visalisation simple leniear regrission
plt.scatter(X, y, c='m')
plt.plot(X, lin_reg.predict(X),color='yellow')
plt.title('MOMO')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

#visalisation poly regrission
X_grid = np.arange(min(X), max(X)+0.1,0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, c='m')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='yellow')
plt.title('MOMO')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

#Predict with the linear model 
lin_reg.predict([[6.5]])

#Predict with the poly model 
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
