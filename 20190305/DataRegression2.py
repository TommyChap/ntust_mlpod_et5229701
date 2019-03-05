# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rows = 3
cols = 3

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Visualising the Linear Regression results
plt.subplot(rows, cols, 1)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Trurh or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

print('score=%.4f' % lin_reg.score(x, y))

# Fitting Ploynomial Regression to the dataset order=2
from sklearn.preprocessing import PolynomialFeatures
poly_reg_2 = PolynomialFeatures(degree = 2)
x_poly_2 = poly_reg_2.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly_2, y)

# Visualising the Polynomal Regression results
plt.subplot(rows, cols, 3)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg_2.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

print('score=%.4f' % lin_reg_2.score(x_poly_2, y))

# Fitting Ploynomial Regression to the dataset order=3
poly_reg_3 = PolynomialFeatures(degree = 3)
x_poly_3 = poly_reg_3.fit_transform(x)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(x_poly_3, y)

# Visualising the Polynomal Regression results
plt.subplot(rows, cols, 7)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_3.predict(poly_reg_3.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

print('score=%.4f' % lin_reg_3.score(x_poly_3, y))

# Fitting Ploynomial Regression to the dataset order=4
poly_reg_4 = PolynomialFeatures(degree = 4)
x_poly_4 = poly_reg_4.fit_transform(x)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(x_poly_4, y)

# Visualising the Polynomal Regression results
plt.subplot(rows, cols, 9)
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg_2.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

print('score=%.4f' % lin_reg_4.score(x_poly_4, y))

# Show results
plt.show()
