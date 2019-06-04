# Pima Indians Diabetes Artificial Neural Network

# Importing the libraries
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dataset1 = np.loadtxt("pima-indians-diabetes.csv", delimiter=',')

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]

def create_default_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


model = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers = ['rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
param_grid = dict(optimizer=optimizers, epochs=epochs,
                  batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
print("test score, param:", grid_result.best_score_, grid_result.best_params_)