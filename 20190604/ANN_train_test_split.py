# Pima Indians Diabetes Artificial Neural Network

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')

inputList = dataset1[:, 0:-1]
resultList = dataset1[:, 8]

feature_train, feature_test, label_train, label_test = \
train_test_split(inputList, resultList, test_size = 0.2)

model = Sequential()

model.add(Dense(units = 12, activation = 'relu', input_dim = 8))
model.add(Dense(units = 8, activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(feature_train, label_train, 
          validation_data = (feature_test, label_test), batch_size = 10, epochs = 150, validation_split = 0.2)
scores = model.evaluate(feature_test, label_test)

print(model.metrics_names[1], scores[1] * 100)
