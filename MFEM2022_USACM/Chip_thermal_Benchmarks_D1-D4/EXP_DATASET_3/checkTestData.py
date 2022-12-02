import numpy as np 
from tqdm import tqdm
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# y_train = np.random.random(size=(1, 80, 80))
# y_test = np.random.random(size=(1, 80, 80))
print('y_train, y_test: ', y_train.shape, y_test.shape)
res = []
# y_train = np.concatenate((y_train, y_test[:10]), axis=0)
for y in tqdm(y_test):
	check = np.all(y_train == y, axis=(1, 2))
	s = sum(check)
	res.append(s)
print('These many exists: ', sum(res))
# 	raise
# print('Check whether y_test is in y_train: ', np.all(y_train == y_test, axis=(1, 2)))