
from sklearn import linear_model
import matplotlib.pyplot as plt

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
reg1 = linear_model.LinearRegression()
reg1.fit(features, values)
print("reg1 coef=", reg1.coef_)
print("reg1 intercept=", reg1.intercept_)

# plt.scatter(features, values, c='green')
# range1 = [-1, 3]
# plt.plot(range1, reg1.coef_[0] * range1 + reg1.coef_[1] * range1 + reg1.intercept_, c='red')
# plt.show()

print("first coef=", reg1.coef_[0], ", second coef=", reg1.coef_[1])
