from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras import backend as K

from utils import LRN2D,conv2d_bn


def create_model():
    myInput = Input(shape=(96, 96, 3))

    x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Lambda(LRN2D, name='lrn_1')(x)
    x = Conv2D(64, (1, 1), name='conv2')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(192, (3, 3), name='conv3')(x)
    x = BatchNormalization(axis=3, epsilon=0.00001, name='bn3')(x)
    x = Activation('relu')(x)
    x = Lambda(LRN2D, name='lrn_2')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)

    # Inception3a
    CNNmodel_3a_3x3 = Conv2D(96, (1, 1), name='CNNmodel_3a_3x3_conv1')(x)
    CNNmodel_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_3x3_bn1')(CNNmodel_3a_3x3)
    CNNmodel_3a_3x3 = Activation('relu')(CNNmodel_3a_3x3)
    CNNmodel_3a_3x3 = ZeroPadding2D(padding=(1, 1))(CNNmodel_3a_3x3)
    CNNmodel_3a_3x3 = Conv2D(128, (3, 3), name='CNNmodel_3a_3x3_conv2')(CNNmodel_3a_3x3)
    CNNmodel_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_3x3_bn2')(CNNmodel_3a_3x3)
    CNNmodel_3a_3x3 = Activation('relu')(CNNmodel_3a_3x3)

    CNNmodel_3a_5x5 = Conv2D(16, (1, 1), name='CNNmodel_3a_5x5_conv1')(x)
    CNNmodel_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_5x5_bn1')(CNNmodel_3a_5x5)
    CNNmodel_3a_5x5 = Activation('relu')(CNNmodel_3a_5x5)
    CNNmodel_3a_5x5 = ZeroPadding2D(padding=(2, 2))(CNNmodel_3a_5x5)
    CNNmodel_3a_5x5 = Conv2D(32, (5, 5), name='CNNmodel_3a_5x5_conv2')(CNNmodel_3a_5x5)
    CNNmodel_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_5x5_bn2')(CNNmodel_3a_5x5)
    CNNmodel_3a_5x5 = Activation('relu')(CNNmodel_3a_5x5)

    CNNmodel_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
    CNNmodel_3a_pool = Conv2D(32, (1, 1), name='CNNmodel_3a_pool_conv')(CNNmodel_3a_pool)
    CNNmodel_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_pool_bn')(CNNmodel_3a_pool)
    CNNmodel_3a_pool = Activation('relu')(CNNmodel_3a_pool)
    CNNmodel_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(CNNmodel_3a_pool)

    CNNmodel_3a_1x1 = Conv2D(64, (1, 1), name='CNNmodel_3a_1x1_conv')(x)
    CNNmodel_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3a_1x1_bn')(CNNmodel_3a_1x1)
    CNNmodel_3a_1x1 = Activation('relu')(CNNmodel_3a_1x1)

    CNNmodel_3a = concatenate([CNNmodel_3a_3x3, CNNmodel_3a_5x5, CNNmodel_3a_pool, CNNmodel_3a_1x1], axis=3)

    # Inception3b
    CNNmodel_3b_3x3 = Conv2D(96, (1, 1), name='CNNmodel_3b_3x3_conv1')(CNNmodel_3a)
    CNNmodel_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_3x3_bn1')(CNNmodel_3b_3x3)
    CNNmodel_3b_3x3 = Activation('relu')(CNNmodel_3b_3x3)
    CNNmodel_3b_3x3 = ZeroPadding2D(padding=(1, 1))(CNNmodel_3b_3x3)
    CNNmodel_3b_3x3 = Conv2D(128, (3, 3), name='CNNmodel_3b_3x3_conv2')(CNNmodel_3b_3x3)
    CNNmodel_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_3x3_bn2')(CNNmodel_3b_3x3)
    CNNmodel_3b_3x3 = Activation('relu')(CNNmodel_3b_3x3)

    CNNmodel_3b_5x5 = Conv2D(32, (1, 1), name='CNNmodel_3b_5x5_conv1')(CNNmodel_3a)
    CNNmodel_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_5x5_bn1')(CNNmodel_3b_5x5)
    CNNmodel_3b_5x5 = Activation('relu')(CNNmodel_3b_5x5)
    CNNmodel_3b_5x5 = ZeroPadding2D(padding=(2, 2))(CNNmodel_3b_5x5)
    CNNmodel_3b_5x5 = Conv2D(64, (5, 5), name='CNNmodel_3b_5x5_conv2')(CNNmodel_3b_5x5)
    CNNmodel_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_5x5_bn2')(CNNmodel_3b_5x5)
    CNNmodel_3b_5x5 = Activation('relu')(CNNmodel_3b_5x5)

    CNNmodel_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(CNNmodel_3a)
    CNNmodel_3b_pool = Conv2D(64, (1, 1), name='CNNmodel_3b_pool_conv')(CNNmodel_3b_pool)
    CNNmodel_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_pool_bn')(CNNmodel_3b_pool)
    CNNmodel_3b_pool = Activation('relu')(CNNmodel_3b_pool)
    CNNmodel_3b_pool = ZeroPadding2D(padding=(4, 4))(CNNmodel_3b_pool)

    CNNmodel_3b_1x1 = Conv2D(64, (1, 1), name='CNNmodel_3b_1x1_conv')(CNNmodel_3a)
    CNNmodel_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name='CNNmodel_3b_1x1_bn')(CNNmodel_3b_1x1)
    CNNmodel_3b_1x1 = Activation('relu')(CNNmodel_3b_1x1)

    CNNmodel_3b = concatenate([CNNmodel_3b_3x3, CNNmodel_3b_5x5, CNNmodel_3b_pool, CNNmodel_3b_1x1], axis=3)

    # Inception3c
    CNNmodel_3c_3x3 = conv2d_bn(CNNmodel_3b,
                                       layer='CNNmodel_3c_3x3',
                                       cv1_out=128,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))

    CNNmodel_3c_5x5 = conv2d_bn(CNNmodel_3b,
                                       layer='CNNmodel_3c_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))

    CNNmodel_3c_pool = MaxPooling2D(pool_size=3, strides=2)(CNNmodel_3b)
    CNNmodel_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(CNNmodel_3c_pool)

    CNNmodel_3c = concatenate([CNNmodel_3c_3x3, CNNmodel_3c_5x5, CNNmodel_3c_pool], axis=3)

    #CNNmodel 4a
    CNNmodel_4a_3x3 = conv2d_bn(CNNmodel_3c,
                                       layer='CNNmodel_4a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=192,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    CNNmodel_4a_5x5 = conv2d_bn(CNNmodel_3c,
                                       layer='CNNmodel_4a_5x5',
                                       cv1_out=32,
                                       cv1_filter=(1, 1),
                                       cv2_out=64,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(1, 1),
                                       padding=(2, 2))

    CNNmodel_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(CNNmodel_3c)
    CNNmodel_4a_pool = conv2d_bn(CNNmodel_4a_pool,
                                        layer='CNNmodel_4a_pool',
                                        cv1_out=128,
                                        cv1_filter=(1, 1),
                                        padding=(2, 2))
    CNNmodel_4a_1x1 = conv2d_bn(CNNmodel_3c,
                                       layer='CNNmodel_4a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    CNNmodel_4a = concatenate([CNNmodel_4a_3x3, CNNmodel_4a_5x5, CNNmodel_4a_pool, CNNmodel_4a_1x1], axis=3)

    #CNNmodel4e
    CNNmodel_4e_3x3 = conv2d_bn(CNNmodel_4a,
                                       layer='CNNmodel_4e_3x3',
                                       cv1_out=160,
                                       cv1_filter=(1, 1),
                                       cv2_out=256,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(2, 2),
                                       padding=(1, 1))
    CNNmodel_4e_5x5 = conv2d_bn(CNNmodel_4a,
                                       layer='CNNmodel_4e_5x5',
                                       cv1_out=64,
                                       cv1_filter=(1, 1),
                                       cv2_out=128,
                                       cv2_filter=(5, 5),
                                       cv2_strides=(2, 2),
                                       padding=(2, 2))
    CNNmodel_4e_pool = MaxPooling2D(pool_size=3, strides=2)(CNNmodel_4a)
    CNNmodel_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(CNNmodel_4e_pool)

    CNNmodel_4e = concatenate([CNNmodel_4e_3x3, CNNmodel_4e_5x5, CNNmodel_4e_pool], axis=3)

    #CNNmodel5a
    CNNmodel_5a_3x3 = conv2d_bn(CNNmodel_4e,
                                       layer='CNNmodel_5a_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))

    CNNmodel_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(CNNmodel_4e)
    CNNmodel_5a_pool = conv2d_bn(CNNmodel_5a_pool,
                                        layer='CNNmodel_5a_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1),
                                        padding=(1, 1))
    CNNmodel_5a_1x1 = conv2d_bn(CNNmodel_4e,
                                       layer='CNNmodel_5a_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))

    CNNmodel_5a = concatenate([CNNmodel_5a_3x3, CNNmodel_5a_pool, CNNmodel_5a_1x1], axis=3)

    #CNNmodel_5b
    CNNmodel_5b_3x3 = conv2d_bn(CNNmodel_5a,
                                       layer='CNNmodel_5b_3x3',
                                       cv1_out=96,
                                       cv1_filter=(1, 1),
                                       cv2_out=384,
                                       cv2_filter=(3, 3),
                                       cv2_strides=(1, 1),
                                       padding=(1, 1))
    CNNmodel_5b_pool = MaxPooling2D(pool_size=3, strides=2)(CNNmodel_5a)
    CNNmodel_5b_pool = conv2d_bn(CNNmodel_5b_pool,
                                        layer='CNNmodel_5b_pool',
                                        cv1_out=96,
                                        cv1_filter=(1, 1))
    CNNmodel_5b_pool = ZeroPadding2D(padding=(1, 1))(CNNmodel_5b_pool)

    CNNmodel_5b_1x1 = conv2d_bn(CNNmodel_5a,
                                       layer='CNNmodel_5b_1x1',
                                       cv1_out=256,
                                       cv1_filter=(1, 1))
    CNNmodel_5b = concatenate([CNNmodel_5b_3x3, CNNmodel_5b_pool, CNNmodel_5b_1x1], axis=3)

    av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(CNNmodel_5b)
    reshape_layer = Flatten()(av_pool)
    dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

    return Model(inputs=[myInput], outputs=norm_layer)
