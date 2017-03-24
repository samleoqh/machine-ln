# Finetune ResNet50, Tensorflow as backend
# Tested on my own dataset polyp (2 classes) and cervix(3 classes)

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image

K.set_image_dim_ordering('tf')

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3  # 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3  # 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def resnet50_model(img_rows, img_cols, color_type=1, num_class=3):
    """
    Resnet Model for Keras
    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_class - number of class labels for our classification task
    """

    bn_axis = 3
    # img_input = Input(shape=(color_type, img_rows, img_cols))
    img_input = Input(shape=(img_rows, img_cols, color_type))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # dim_ordering='th'

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    # x_fc = Flatten()(x_fc)
    # x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x)

    # Load ImageNet pre-trained data
    # weights_path = './model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = './model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    # x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten(input_shape=model.output_shape[1:])(x)
    x_newfc = Dense(num_class, activation='sigmoid', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)
    model.summary()
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9) #sparse_categorical_crossentropy
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import argparse
import cv2
import os


def image_to_feature_vector(image, size=(224, 224)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def load_data(imgTrainPath, imgTestPath):
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    X_train = []
    Y_train = []
    X_valid = []
    Y_valid = []

    for (i, imagePath) in enumerate(imgTrainPath):
        # load the image and extract the class label
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)

        # update the raw images, features, and labels matricies,
        # respectively
        X_train.append(pixels)
        Y_train.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed train images{}/{}".format(i, len(imgTrainPath)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    X_train = np.array(X_train)


    for (i, imagePath) in enumerate(imgTestPath):
        # load the image and extract the class label
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image)

        # update the raw images, features, and labels matricies,
        # respectively
        X_valid.append(pixels)
        Y_valid.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed validation images{}/{}".format(i, len(imgTestPath)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    Y_valid = le.fit_transform(Y_valid)
    X_valid = np.array(X_valid)

    return X_train, X_valid, Y_train, Y_valid

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Fine-tune Example
    img_rows, img_cols = 224, 224  # Resolution of inputs
    color_type = 3
    channel = 3
    num_class = 3
    batch_size = 16
    nb_epoch = 8

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", type=str, default='cervix2/train',
                    help="path to input dataset")
    ap.add_argument("-v", "--test", type=str, default='cervix2/test',
                    help="path to input dataset")

    args = vars(ap.parse_args())

    # grab the list of images that we'll be describing
    print("[INFO] describing images...")
    imgTrainPath = list(paths.list_images(args["train"]))
    imgTestPath = list(paths.list_images(args["test"]))

    X_train, X_valid, Y_train, Y_valid = load_data(imgTrainPath, imgTestPath)
    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_class)

    # Start Fine-tuning
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, color_type)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, color_type)
    Y_train = np_utils.to_categorical(Y_train,num_class)
    Y_valid = np_utils.to_categorical(Y_valid,num_class)

    history=model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              validation_data=(X_valid, Y_valid),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.xticks(np.arange(0, 100, 2))
    plt.legend(['train', 'test'], loc='upper left')

    plt.grid(True)

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, 100, 2))
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)

    plt.show()
