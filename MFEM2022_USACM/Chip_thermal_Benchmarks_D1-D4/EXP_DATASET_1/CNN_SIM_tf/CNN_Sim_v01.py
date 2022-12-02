import os
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ReLU, Add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
import numpy as np
# import pyvista as pv
import numpy as np
import glob
# import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import RegularGridInterpolator
import shutil
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

cur_path = os.getcwd()
os.makedirs('./CNN_Sim', mode=0o777, exist_ok=True)
work_dir = os.getcwd() + '/CNN_Sim/'
os.makedirs(work_dir + 'ckpt', mode=0o777, exist_ok=True)
ckpt_dir = work_dir + 'ckpt/'

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




# Data
def get_data():
    print('==> Preparing data..')
    val_percent = 0.1
    batch_size = 100

    dir_data = './data/'
    train_x = np.load(dir_data + 'x_train.npy').astype(np.float32)
    train_y = np.load(dir_data + 'y_train.npy').astype(np.float32)

    ##### Filter unrealistic cases #####
    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    ##### Filter unrealistic cases #####

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_y = np.expand_dims(train_y, -1)

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    # raise

    test_x = np.load(dir_data + 'x_test.npy').astype(np.float32)
    test_y = np.load(dir_data + 'y_test.npy').astype(np.float32)

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]

    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_y = np.expand_dims(test_y, -1)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)

    return train_x, train_y, test_x, test_y


def plot(CNN_SIM, x_test, y_test, T_min, T_max):
    idx = np.random.choice(x_test.shape[0])

    pred = CNN_SIM(x_test[idx:idx+1, :, :, :])
    true = y_test

    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min

    fig = plt.figure(figsize=(15,3))
    plt.subplots_adjust(wspace=0.1)
    ax = fig.add_subplot(131)
    im = ax.imshow(pred[0, :, :], cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title('Prediction')
    plt.colorbar(im,cax=cax)

    ax = fig.add_subplot(132)
    im = ax.imshow(true[idx, :, :], cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title('True')
    plt.colorbar(im,cax=cax)

    ax = fig.add_subplot(133)
    im = ax.imshow((y_test[idx, :, :] - pred[0, :, :]), cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title('Abs error')
    plt.colorbar(im,cax=cax)
    plt.savefig('pred.jpg')

def mape(true, pred, T_min, T_max): 
    true, pred = np.array(true), np.array(pred)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    mask = true != 0
    return np.mean(np.abs((true - pred) / true)[mask])

def relativeL2(true, pred, T_min, T_max):
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    return np.linalg.norm(true.flatten() - pred.numpy().flatten()) / np.linalg.norm(true.flatten()) 
    
def mae(true, pred, T_min, T_max):
    print('pred true', pred.shape, true.shape)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))

def calculate_accuracy(x_test, y_test, T_min, T_max):
    true = y_test
    pred = CNN_SIM(x_test)
    mae_error = mae(true, pred, T_min, T_max)
    mape_error = mape(true, pred, T_min, T_max)
    rel_l2 = relativeL2(true, pred, T_min, T_max)
    print('mae: ', mae_error)
    print('rel_l2: ', rel_l2)
    print('mape_error: ', mape_error)
    with open('./final_test_l2.txt', 'w') as f:
        f.write(f'mae is: {mae} \n')
        f.write(f'relative l2 is: {rel_l2} \n')
        f.write(f'mape_error is: {mape_error} \n')

x_train, y_train, x_test, y_test = get_data()
T_max = y_train.max()
T_min = y_train.min()
test_idx = np.random.choice(x_train.shape[0], x_train.shape[0]//10, replace=False)
train_idx = np.setxor1d(np.arange(x_train.shape[0]), test_idx)



my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(ckpt_dir + 'model.hdf5',
                                       save_weights_only=True,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True, verbose=True),
    tf.keras.callbacks.TensorBoard(log_dir = work_dir + '\\logs'),
]

input_size = 80
num_filter = 48
dCNN_dilations = [1, 2, 4, 8, 4, 2, 1]
optimizer = Adam(lr=0.0003)

CNN_SIM = CNN_sim(input_size, num_filter, dCNN_dilations)
CNN_SIM.compile(optimizer=optimizer, loss='mse')


load_model = True
# EPOCH = 1250*2
EPOCH = 5000
if load_model:
    CNN_SIM.built = True
    CNN_SIM.load_weights(ckpt_dir + 'model.hdf5')
    plot(CNN_SIM, x_test, y_test, T_min, T_max)
    calculate_accuracy(x_test, y_test, T_min, T_max)
    # plot(CNN_SIM, x_train, y_train)
    raise
    # tf.keras.models.load_model()

    
history = CNN_SIM.fit(x_train[train_idx, :,:,:], y_train[train_idx, :,:,:],
                epochs=EPOCH,
                batch_size=128,
                shuffle=True,
                verbose=True,
                validation_split=0.2,
                callbacks=my_callbacks)


train_loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(train_loss)
ax.semilogy(val_loss)
ax.legend(['train loss', 'val loss'])
plt.savefig(work_dir + 'loss_dilation')
plot(CNN_SIM, x_test, y_test, T_min, T_max)
















