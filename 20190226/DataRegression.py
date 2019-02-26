from sklearn import linear_model
import matplotlib.pyplot as plt

reg1 = linear_model.LinearRegression()
features = [[1], [2], [3]]
values = [1, 4, 5.5]
plt.scatter(features, values, c='green')

# y = ax+b, a ==> coef, b ==> intercept
reg1.fit(features, values)
print('coef=%f, intercept=%f' %(reg1.coef_, reg1.intercept_))

range1 = [-1, 3]
plt.show(plt.plot(range1, reg1.coef_ * range1 + reg1.intercept_, c='gray'))
