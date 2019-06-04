# Pima Indians Diabetes Artificial Neural Network

# Importing the libraries
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

dataset1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

def create_default_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    # 把node用edge連接起來
    # loss=binary_crossentropy使用於二分法
    # adam隨機梯度下降的最佳化參數
    # accuracy測量的指標
    return model


model = KerasClassifier(build_fn=create_default_model, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, inputList, resultList, cv=fiveFold)
print("mean = %.3f, std = %.3f" % (results.mean(), results.std()))