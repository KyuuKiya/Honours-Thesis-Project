import numpy as np

from keras.models import *
from keras.layers import *
from keras.optimizers import *

get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
from segmentation_models.losses import *
from segmentation_models.metrics import *

def vnet(input_size = (128, 128, 64, 1), optimizer = Adam(lr=1e-4),
         loss = dice_loss, metrics = [iou_score]):
    
    inputs = Input(input_size)
    conv1 = Conv3D(16, 5, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    add1 = Add()([inputs, conv1])
    down1 =  Conv3D(32, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add1)
    
    conv2 = Conv3D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(down1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv3D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    add2 = Add()([down1, conv2])
    down2 =  Conv3D(64, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add2)
    
    conv3 = Conv3D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(down2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv3D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    add3 = Add()([down2, conv3])
    down3 =  Conv3D(128, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add3)
    
    conv4 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(down3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    add4 = Add()([down3, conv4])
    up4 =  Conv3DTranspose(128, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add4)
    
    concat5 = concatenate([add3,up4])
    conv5 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(concat5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv3D(128, 5, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    add5 = Add()([up4, conv5])
    up5 =  Conv3DTranspose(64, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add5)
    
    concat6 = concatenate([add2,up5])
    conv6 = Conv3D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(concat6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    conv6 = Conv3D(64, 5, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)
    add6 = Add()([up5, conv6])
    up6 =  Conv3DTranspose(32, 2, strides=(2, 2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(add6)
    
    concat7 = concatenate([add1,up6])
    conv7 = Conv3D(32, 5, padding = 'same', kernel_initializer = 'he_normal')(concat7)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)
    add7 = Add()([up6, conv7])
    
    conv8 = Conv3D(1, 1, activation = 'sigmoid')(add7)

    model = Model(inputs = [inputs], outputs = [conv8])

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    return model




