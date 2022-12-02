
import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

print('x_train.max(): ', x_train.max())
print('y_train.max(): ', y_train.max())

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print('x_test.max(): ', x_test.max())
print('y_test.max(): ', y_test.max())