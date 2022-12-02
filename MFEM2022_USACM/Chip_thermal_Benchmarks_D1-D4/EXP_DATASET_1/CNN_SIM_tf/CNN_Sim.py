import os
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ReLU, Add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import numpy as np
import glob
import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import shutil
import re

cur_path = os.getcwd()
os.makedirs('./CNN_Sim', mode=0o777, exist_ok=True)
work_dir = os.getcwd() + '/CNN_Sim/'
os.makedirs(work_dir + 'ckpt', mode=0o777, exist_ok=True)
ckpt_dir = work_dir + 'ckpt/'

input_files = glob.glob('.\input\density_structured_*.npy')
output_files_X = glob.glob('.\output\DispX*.vtk.npy')
output_files_Y = glob.glob('.\output\DispY*.vtk.npy')
output_files_Z = glob.glob('.\output\DispZ*.vtk.npy')
x_coord_51 = np.load('.\\output\\x_Structured.npy')
y_coord_51 = np.load('.\\output\\y_Structured.npy')
z_coord_51 = np.load('.\\output\\z_Structured.npy')
# output_files = glob.glob('.\output\*.vtu')
# mesh = input_files[0]
# density = meshio.read(input_files[1]).cell_data['density'][0]
# keys = list(meshio.read(output_files[1]).point_data.keys())
# x_disp = meshio.read(output_files[1]).point_data[keys[0]]
# mesh = pv.read('layerout_L000001.vtk')
# print('density shape is:' , density.shape)
# print('x_disp shape is:' , x_disp.shape)
x_50, y_50, z_50 = np.linspace(0, 10, 50), np.linspace(0, 10, 50), np.linspace(0, 10, 50)
x_51, y_51, z_51 = np.linspace(0, 10, 51), np.linspace(0, 10, 51), np.linspace(0, 10, 51)
# x_coord, y_coord, z_coord = np.meshgrid(xx, yy, zz)

img_size = 64
xxx, yyy, zzz = np.linspace(0, 10, img_size), np.linspace(0, 10, img_size), np.linspace(0, 10, img_size)
globals()[f'x_coord_{img_size}'], globals()[f'y_coord_{img_size}'], globals()[f'z_coord_{img_size}'] = np.meshgrid(xxx, yyy, zzz)
output_np = np.empty((0, img_size,img_size,img_size))
input_np = np.empty((0, img_size,img_size,img_size))

# dir_input_renamed = '.\\input\\renamed\\'
# os.makedirs(dir_input_renamed, mode=0o777, exist_ok=True)
# for input_file in input_files:
#     file_number = re.findall(r'\d+', input_file.split('\\')[-1])[0]
#     number_str = str(file_number)
#     zero_filled_number = number_str.zfill(6)
#     new_name = 'Density_L' + zero_filled_number + '.npy'
#     orig_data = np.load(input_file)
#     np.save(dir_input_renamed + new_name, orig_data)

renamed_input_files = glob.glob('.\\input\\renamed\\Density_L*.npy')
for input_file, output_file in zip(renamed_input_files, output_files_Z):
    input_data = np.load(input_file)
    output_data = np.load(output_file)

    # print('input data shape is: ', input_data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(x_coord.flatten(), y_coord.flatten(), z_coord.flatten(), c = input_data.flatten(), cmap='jet')
    # ax.scatter(x_coord_51.flatten(), y_coord_51.flatten(), z_coord_51.flatten(), c=output_data.flatten(), cmap='jet')
    # plt.show()

    interpolating_function_input = RegularGridInterpolator((x_50, y_50, z_50), input_data)
    interpolating_function_output = RegularGridInterpolator((x_51, y_51, z_51), output_data)
    new_coords = np.hstack((globals()[f'x_coord_{img_size}'].reshape(-1, 1),\
                            globals()[f'y_coord_{img_size}'].reshape(-1, 1),\
                            globals()[f'z_coord_{img_size}'].reshape(-1, 1)))
    input_interp = interpolating_function_input(new_coords).reshape((1, img_size, img_size, img_size))
    output_interp = interpolating_function_output(new_coords).reshape((1, img_size, img_size, img_size))
    input_np = np.concatenate((input_np, input_interp), axis=0)
    output_np = np.concatenate((output_np, output_interp), axis=0)
print('done')

# latent_size = 16
class CNN_sim(Model):
    def __init__(self, input_size, num_filter, dCNN_dilations):
        super(CNN_sim, self).__init__()
        self.input_size = input_size
        self.num_filter = num_filter
        self.dCNN_dilations = dCNN_dilations
        self.activation = 'relu'
        
        self.encoder = self.encoder_net()
        self.decoder = self.decoder_net()
        self.processor = self.processor_net()
    
    def conv_Block(self, x, BN, activation, filters):
        for i in range(1):
            x = Conv2D(filters=int(filters), kernel_size = (3, 3), activation=activation, padding='same')(x)
        if BN:
            x = BatchNormalization()(x)
        return x

    def residual_block(self, x, filters, dilation_rates):
#         print('x_shortcut is: ', x)
        y = Conv2D(filters=filters, kernel_size=(3,3), padding='same', dilation_rate=dilation_rates[0])(x)
        y = ReLU()(y)
        y = Conv2D(filters=filters, kernel_size=(3,3), padding='same', dilation_rate=dilation_rates[1])(y)
        y = ReLU()(y)
        out = Add()([x, y])
#         print('after residual block, out is: ', out)
        out = ReLU()(out)
        return out

    def grouped(self, iterable, n):
        "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
        return zip(*[iter(iterable)]*n)
    
    def encoder_net(self):
        BN = False
        inputs = Input(shape=(self.input_size, self.input_size, 1), name='Encoder_input')
        x = inputs
        for i in range(2):  
            x = self.conv_Block(x, BN, None, self.num_filter)
            x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
        encoder = Model(inputs=[inputs], outputs=[x])
        return encoder
    
    def decoder_net(self):
        BN = False
        inputs = Input(shape=(self.input_size//4, self.input_size//4, self.num_filter), name='Decoder_input')
        x = inputs
        for i in range(2):
            x = self.conv_Block(x, BN, None, self.num_filter)
            x = UpSampling2D()(x)
        x = Conv2D(filters=1, kernel_size=(3,3), padding='same')(x)
        decoder = Model(inputs=[inputs], outputs=[x])
        return decoder
    
    def processor_net(self):
        inputs = Input(shape=(self.input_size//4, self.input_size//4,num_filter))
        x = inputs
        x = Conv2D(filters=int(num_filter), kernel_size=(3,3), padding='same', dilation_rate=1)(x)
        x = ReLU()(x)
        for dCNN_dilation_1, dCNN_dilation_2 in self.grouped(self.dCNN_dilations[1:], 2):
            dCNN_dilation = [dCNN_dilation_1, dCNN_dilation_2]
#             print('dCNN_dilation is: ', dCNN_dilation)
            x = self.residual_block(x, num_filter, dCNN_dilation)
        processor = Model(inputs=[inputs], outputs=[x])
        return processor
    
    def call(self, x):
#         print('x shape is: ', x)
        x = self.encoder(x)
#         print('After encoding, x shape is: ', x)
        x = self.processor(x)
#         print('After processing, x shape is: ', x)
        x = self.decoder(x)
#         print('After decoding, x shape is: ', x)
        return x
        
#     def net(self):
#         net = Sequential([self.encoder, self.processor, self.decoder])
#         return net
    
    def encode(self, x):
        x = self.encoder(x)
        return x        

x_train = input_np
y_train = output_np

input_max = x_train.max()
input_min = x_train.min()
x_train = (x_train - input_min) / (input_max - input_min)

output_max = y_train.max()
output_min = y_train.min()
y_train = (y_train - output_min) / (output_max - output_min)

test_idx = np.random.choice(x_train.shape[0], x_train.shape[0]//10, replace=False)
train_idx = np.setxor1d(np.arange(x_train.shape[0]), test_idx)

my_callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir + 'model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir = work_dir + '/logs'),
]

input_size = 64
num_filter = 48
dCNN_dilations = [1, 2, 4, 8, 4, 2, 1]
optimizer = Adam(lr=0.0003)

CNN_SIM = CNN_sim(input_size, num_filter, dCNN_dilations)
CNN_SIM.compile(optimizer=optimizer, loss='mse')
history = CNN_SIM.fit(x_train[train_idx, :,:,:], x_train[train_idx, :,:,:],
                epochs=1000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_train[test_idx, :,:,:], x_train[test_idx, :,:,:]),
                callbacks=my_callbacks)

# history = autoencoder.fit(x_train[train_idx, :,:,:], x_train[train_idx, :,:,:],
#                 epochs=5,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_train[test_idx, :,:,:], x_train[test_idx, :,:,:]),
#                 callbacks=my_callbacks)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(train_loss)
ax.semilogy(val_loss)
ax.legend(['train loss', 'val loss'])
plt.savefig(work_dir + 'loss')



from mpl_toolkits.axes_grid1 import make_axes_locatable


idx = np.random.choice(x_train.shape[0])

pred = CNN_SIM(x_train[idx:idx+1, :, :, :])

fig = plt.figure(figsize=(15,3))
plt.subplots_adjust(wspace=0.1)
ax = fig.add_subplot(131)
im = ax.imshow(pred[0, :, :])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

ax = fig.add_subplot(132)
im = ax.imshow(x_train[idx, :, :])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

ax = fig.add_subplot(133)
im = ax.imshow((x_train[idx, :, :] - pred[0, :, :]))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)
