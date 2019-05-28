# Pima Indians Diabetes Artificial Neural Network

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')

inputList = dataset1[:, 0:-1]
resultList = dataset1[:, 8]
output = []

model = Sequential()

model.add(Dense(units = 12, activation = 'relu', input_dim = 8))
model.add(Dense(units = 4, activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(inputList, resultList, batch_size = 10, epochs = 150, validation_split = 0.2)
scores = model.evaluate(inputList, resultList)

print(model.metrics_names[1], scores[1] * 100)
output.append((model.metrics_names[1], scores[1] * 100))
