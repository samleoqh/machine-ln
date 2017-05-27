# Author: Qinghui Liu @ USN 2017-04-25
# For master thesis project
# Data input generation and model histories plot function
# Score mean merge function
# Create CSV for kaggle submission
# Need to reflect and refine the source code in future
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")
from keras import backend as K
K.set_image_dim_ordering('tf')  #for ResNet50 tensorflow backend
#K.set_image_dim_ordering('th') #for VGG16


def data_input_gen(tf_th='tf', size=(224, 224), batch = 8,
                   train_dir= './polyp2/train',
                   validation_dir= './polyp2/test'):

    K.set_image_dim_ordering(tf_th)
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=size,
        batch_size=batch,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=size,
        batch_size=batch,
        class_mode='categorical')
    return train_generator,validation_generator


def histories_plt(histories,cyc_color,title = 'histories'):
    # dict_keys(
    # ['precision', 'mean_squared_error', 'acc',
    # 'val_fmeasure', 'fmeasure', 'val_mean_squared_error',
    # 'val_loss', 'loss',
    # 'val_precision', 'val_acc',
    # 'val_recall', 'recall']
    # )
    fig = plt.figure()
    fig.suptitle(title,fontsize=14,fontweight='bold')

    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    hist_num = len(histories)
    ax0.set_color_cycle(cyc_color)

    ax0.set_title('Models Accuracy')
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Epoch')
    ax0.set_yticks(np.arange(0.4, 1.05, 0.05))
    ax0.set_xticks(np.arange(0, 500, 5))

    ax0.grid(True, linestyle=':')
    for i in range(hist_num):
        history = histories[i]
        ax0.plot(history.history['acc'],'-')
        ax0.plot(history.history['val_acc'],':')

    ax1.set_color_cycle(cyc_color)
    ax1.set_title('Models Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    #ax1.set_yticks(np.arange(0.1, 0.85, 0.05))
    ax1.set_xticks(np.arange(0, 500, 5))
    ax1.grid(True,linestyle=':')
    for i in range(hist_num):
        history = histories[i]
        ax1.plot(history.history['loss'],'-')
        ax1.plot(history.history['val_loss'],':')

    fig.subplots_adjust(hspace = 0.4)

    #plt.savefig('histories.png')
    plt.show()

def image_to_feature_vector(image, height=224, width= 224):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    img = cv2.resize(image, (height,width))
    cv2.normalize(img,img,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    #return cv2.resize(image, size).flatten()
    return img.flatten()


def load_train_data(train_dir,num_class,th_tf='tf',height=224,width=224,dim=3):
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    X_train = []
    Y_train = []
    X_train_id = []
    imgTrainPath = list(paths.list_images(train_dir))

    for (i, imagePath) in enumerate(imgTrainPath):
        # load the image and extract the class label
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        flbase = imagePath.split(os.path.sep)[-1]

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image,height=height,width=width)

        # update the raw images, features, and labels matricies,
        # respectively
        X_train.append(pixels)
        Y_train.append(label)
        X_train_id.append(flbase)

        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed train images{}/{}".format(i, len(imgTrainPath)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    X_train = np.array(X_train, dtype=np.uint8)
    #X_train = X_train.astype('float')
    #X_train = X_train / 255
    X_train_id = np.array(X_train_id)

    #Y_train = np_utils.to_categorical(Y_train, nb_classes=num_class)
    Y_train = np_utils.to_categorical(Y_train, num_classes=num_class)
    if th_tf == 'tf':
        X_train = X_train.reshape(X_train.shape[0], height, width, dim)
    else:
        X_train = X_train.reshape(X_train.shape[0], dim,height, width)


    return X_train, Y_train, X_train_id


def load_test_data(test_dir,num_class, th_tf='tf',height=224,width=224,dim=3):
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    X_valid = []
    Y_label = []
    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    le = LabelEncoder()
    imgTestPath = list(paths.list_images(test_dir))

    for (i, imagePath) in enumerate(imgTestPath):
        # load the image and extract the class label
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-2]

        # extract raw pixel intensity "features", followed by a color
        # histogram to characterize the color distribution of the pixels
        # in the image
        pixels = image_to_feature_vector(image,height=height,width=width)

        # update the raw images, features, and labels matricies,
        # respectively
        X_valid.append(pixels)
        Y_label.append(label)

        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed test images{}/{}".format(i, len(imgTestPath)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    Y_valid = le.fit_transform(Y_label)
    Y_label = np.array(Y_label)
    X_valid = np.array(X_valid,dtype=np.uint8)
    #X_valid = X_valid.astype('float')
    #X_valid = X_valid/255

    #Y_valid = np_utils.to_categorical(y=Y_valid, nb_classes=num_class)
    Y_valid = np_utils.to_categorical(y=Y_valid,num_classes=num_class)
    if th_tf == 'tf':
        X_valid = X_valid.reshape(X_valid.shape[0], height, width, dim)
    else:
        X_valid = X_valid.reshape(X_valid.shape[0], dim, height, width)

    return X_valid, Y_valid, Y_label


def load_realtest_data(test_dir,th_tf='tf',height=224,width=224,dim=3):
    # initialize the raw pixel intensities matrix, the features matrix,
    # and labels list
    X_test = []
    X_test_id = []
    imgTestPath = list(paths.list_images(test_dir))

    for (i, imagePath) in enumerate(imgTestPath):
        # load the image and extract the class label
        image = cv2.imread(imagePath)
        flbase = imagePath.split(os.path.sep)[-1]

        pixels = image_to_feature_vector(image,height=height,width=width)

        X_test.append(pixels)
        X_test_id.append(flbase)

        # show an update every 1,000 images
        if i > 0 and i % 100 == 0:
            print("[INFO] processed test images{}/{}".format(i, len(imgTestPath)))

    # show some information on the memory consumed by the raw images
    # matrix and features matrix
    X_test = np.array(X_test, dtype=np.uint8)
    #X_test = X_test.astype('float')
    #X_test = X_test / 255
    #X_test_id = np.array(X_test_id)
    if th_tf=='tf':
        X_test = X_test.reshape(X_test.shape[0], height, width, dim)
    else:
        X_test = X_test.reshape(X_test.shape[0], dim, height, width)

    return X_test, X_test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def create_submission(predictions, test_id, info, columns):
    result1 = pd.DataFrame(predictions,columns=columns)
    result1.loc[:, 'image_name'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def histories_plt2(histories,cyc_color,title = 'histories',leg='K'):
    # dict_keys(
    # ['precision', 'mean_squared_error', 'acc',
    # 'val_fmeasure', 'fmeasure', 'val_mean_squared_error',
    # 'val_loss', 'loss',
    # 'val_precision', 'val_acc',
    # 'val_recall', 'recall']
    # )
    fig = plt.figure()
    fig.suptitle(title,fontsize=14,fontweight='bold')

    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    lables = []
    # fig,(ax0,ax1) = plt.subplots(nrows=2)
    hist_num = len(histories)

    for i in range(hist_num):
        lable = leg + '-' + str(i+1)
        lables.append(lable)

    ax0.set_color_cycle(cyc_color)

    ax0.set_title('Models Accuracy')
    ax0.set_ylabel('Accuracy')
    ax0.set_xlabel('Epoch')
    #ax0.set_yticks(np.arange(0.4, 1.05, 0.05))
    ax0.set_xticks(np.arange(0, 500, 20))

    #ax0.grid(True, linestyle=':')
    for i in range(hist_num):
        history = histories[i]
        ax0.plot(history.history['acc'],'-',label=lables[i]+' train')
        ax0.plot(history.history['val_acc'],':',label=lables[i]+' val')

    #ax0.legend(['train', 'test'], loc='upper right')
    #ax0.legend(bbox_to_anchor=(1.05,1.05),
    #           ncol=3, fancybox=True, shadow = True)
    ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
               ncol=3)  # , fancybox=True, shadow=True)


    ax1.set_color_cycle(cyc_color)
    ax1.set_title('Models Error')
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Epoch')
    #ax1.set_yticks(np.arange(0.1, 0.85, 0.05))
    ax1.set_xticks(np.arange(0, 500, 20))
    #ax1.grid(True,linestyle=':')
    for i in range(hist_num):
        history = histories[i]
        ax1.plot(history.history['mean_squared_error'],'-',label=lables[i]+' train')
        ax1.plot(history.history['val_mean_squared_error'],':',label=lables[i]+' val')

    ax1.legend(loc='upper center',bbox_to_anchor=(0.5, 1),
               ncol=3)#, fancybox=True, shadow=True)

    fig.subplots_adjust(hspace = 0.4)

    #plt.savefig('histories.png')
    plt.show()
