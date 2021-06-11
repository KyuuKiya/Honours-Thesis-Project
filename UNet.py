import numpy as np

from keras.models import *
from keras.layers import *
from keras.optimizers import *

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
from segmentation_models.losses import *
from segmentation_models.metrics import *

def unet_3d(input_size = (128, 128, 64, 1), optimizer = Adam(lr=1e-4),
         loss = dice_loss, metrics = [iou_score]):
    
    inputs = Input(input_size)
    conv1 = Conv3D(16, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv3D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv1)
    
    conv2 = Conv3D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)
    
    conv3 = Conv3D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv3)
    
    conv4 = Conv3D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)

    up5 = Conv3DTranspose(512, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    concat5 = concatenate([conv3,up5])
    conv5 = Conv3D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(concat5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(256, 3, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)

    up6 = Conv3DTranspose(256, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    concat6 = concatenate([conv2,up6])
    conv6 = Conv3D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(concat6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv3D(128, 3, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    up7 = Conv3DTranspose(128, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    concat7 = concatenate([conv1,up7])
    conv7 = Conv3D(64, 3, padding = 'same', kernel_initializer = 'he_normal')(concat7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    conv7 = Conv3D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
        
    conv8 = Conv3D(1, 1, activation = 'sigmoid')(conv7)

    model = Model(inputs = [inputs], outputs = [conv8])

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    return model




