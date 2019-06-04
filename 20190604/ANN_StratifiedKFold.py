# Pima Indians Diabetes Artificial Neural Network

# Importing the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

dataset1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')

inputList = dataset1[:, 0:-1]
resultList = dataset1[:, 8]

fiveFold = StratifiedKFold(n_splits = 5, shuffle = True)
totalscores = []

model = Sequential()

model.add(Dense(units = 12, activation = 'relu', input_dim = 8))
model.add(Dense(units = 8, activation = 'relu'))

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['accuracy'])

for train, test in fiveFold.split(inputList, resultList):
    model.fit(inputList[train], resultList[train], 
              validation_data = (inputList[test], resultList[test]), batch_size = 10, epochs = 150, validation_split = 0.2, verbose = 0)
     # 設定model的參數，進行訓練
    scores = model.evaluate(inputList[test], resultList[test])
    # 進行評估
    totalscores.append(scores[1] * 100)

print(totalscores)
