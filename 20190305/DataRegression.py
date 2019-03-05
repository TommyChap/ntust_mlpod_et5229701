# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Training set results
plt.subplot(3, 1, 1)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
#plt.show()

# Visualising the Testing set results
plt.subplot(3, 1, 3)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title('Salary VS Experience (testing set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')

# Show results
plt.show()

print('coef=%f, intercept=%f' %(regressor.coef_, regressor.intercept_))
print('score=%.4f' % regressor.score(x, y))
