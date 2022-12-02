import numpy as np

d1_path = '../3_Distributed_HTC_Unet_lightweight_200um_3000cases/'
d2_path = '../3_Distributed_HTC_Z_NOTUSED_200um_2000CASES/'

x_train = 'x_train.npy'
# x_test = 'x_test.npy'
y_train = 'y_train.npy'
# y_test = 'y_test.npy'
htc_top_train = 'htc_top_train.npy'
htc_btm_train = 'htc_btm_train.npy'
# htc_top_test = 'htc_top_test.npy'
# htc_btm_test = 'htc_btm_test.npy'
# f_names = [x_train,x_test,y_train,y_test, htc_btm_test, htc_btm_train, htc_top_test, htc_top_train]
f_names = [x_train,y_train, htc_btm_train, htc_top_train]
for f in f_names:
	d1 = np.load(d1_path + f)
	d2 = np.load(d2_path + f)
	d3 = np.concatenate((d1, d2), axis=0)
	print(f'{f}: ', d3.shape)
	np.save('./' + f, d3)