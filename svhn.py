
import os
import time
from datetime import timedelta
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model

import sys
import tarfile
import numpy as np
from six.moves.urllib.request import urlretrieve
import scipy.io as sio
from six.moves import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K

import keras
from keras.utils import to_categorical
from keras.preprocessing import image as keras_image

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras import backend
from keras import losses
from keras.metrics import top_k_categorical_accuracy, categorical_accuracy
from keras.engine.topology import Layer
from keras.optimizers import Adam, Nadam
from keras.engine import InputLayer
from keras.models import Sequential, load_model, Model

from keras.layers import Input, BatchNormalization, Flatten, Dropout
from keras.layers import Dense, LSTM, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import UpSampling2D, Conv2DTranspose, DepthwiseConv2D
from keras.layers.core import RepeatVector, Permute
from keras.layers import Reshape, concatenate, merge

from keras import __version__
print('keras version:', __version__)


from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Input, Dense
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.callbacks import TensorBoard
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback





#PRE-Processing --------------------------------------- START
# %matplotlib inline
plt.rcParams['figure.figsize'] = (16.0, 4.0)


#Credit to https://github.com/prafsoni/SVHN for download functions

def main():
    print("Hello World!")


    url = 'http://ufldl.stanford.edu/housenumbers/'
    last_percent_reported = None

    def download_progress_hook(count, blockSize, totalSize):
        global last_percent_reported
        percent = int(count * blockSize * 100 / totalSize)
        
        if last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()
            
            last_percent_reported = percent


    def maybe_download(filename, expected_bytes, force=False, dirc=""):
        d = False
        if force or not os.path.exists(dirc + filename):
            print ('Attempting to download: ' + filename)
            d = True
            filename, _ = urlretrieve(url + filename, dirc + filename, reporthook=download_progress_hook)
            print ('\nDownload Complete!')
        if not d:
            filename = dirc + filename
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print ('Found and verified', filename)
        else:
            raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

    path = 'Input-Data'
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)


    train_file = maybe_download('train_32x32.mat', 182040794, dirc="Input-Data/") #"Input-Data/train_32x32.mat"
    test_file = maybe_download('test_32x32.mat', 64275384, dirc="Input-Data/") #"Input-Data/test_32x32.mat"




    def load_data(path):
        """ Helper function for loading a MAT-File"""
        data = loadmat(path)
        return data['X'], data['y']

    X_train, y_train = load_data('Input-Data/train_32x32.mat')
    X_test, y_test = load_data('Input-Data/test_32x32.mat')

    print("Training Set", X_train.shape, y_train.shape)
    print("Test Set", X_test.shape, y_test.shape)

    #Current shape is: (Width, Height, Channels(coulours), Size)

    # Transposing the image arrays
    #To re-shape the dataframes into: (Size, Width, Height, Channels(Colours))

    X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
    X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

    # Calculate the total number of images
    num_images = X_train.shape[0] + X_test.shape[0]
    print("Total Number of Images", num_images)

    print("Training Set", X_train.shape, y_train.shape)
    print("Test Set", X_test.shape, y_test.shape)

    def plot_images(img, labels, nrows, ncols):
        """ Plot nrows x ncols images
        """
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flat): 
            if img[i].shape == (32, 32, 3):
                ax.imshow(img[i])
            else:
                ax.imshow(img[i,:,:,0])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(labels[i])

    print(np.unique(y_train))

    # Plotting some training set images:
    plot_images(X_train, y_train, 2, 5)

    # Plot some test set images
    plot_images(X_test, y_test, 2, 8)

    #10 is the categorical class for zero. This is confusing and needs to be re-coded





    #Changing 10 to Zero as shown in the above 
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    print(np.unique(y_train))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle('Class Distribution of Digits (0 - 9) ', fontsize=14, fontweight='bold')

    ax1.hist(y_train,  bins=10)
    ax1.set_title("Training set")
    ax1.set_xlim(0, 9)

    ax2.hist(y_test, color='green', bins=10)
    ax2.set_title("Test set")
    ax2.set_xlim(0, 9)

    #We notice there's many 1's and 2's relative to the other digits



    print(np.unique(y_train))

    print("Final Shape of Training Set:", X_train.shape, y_train.shape)
    print("Final Shape of Testing Set:", X_test.shape, y_test.shape)





    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=7)

    #Visualize New Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

    ax1.hist(y_train, bins=9)
    ax1.set_title("Training set")
    ax1.set_xlim(0, 9)

    ax2.hist(y_val, color='g', bins=9)
    ax2.set_title("Validation set")
    ax2.set_xlim(0, 9)

    fig.tight_layout()

    y_train.shape, y_val.shape, y_test.shape
    X_train.shape, X_val.shape, X_test.shape

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    #Grayscale Conversion
    #To speed up our experiments we will convert our images from RGB to Grayscale, 
    #which grately reduces the amount of data we will have to process.

    def rgb2gray(images):
        return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

    #Converting to Float for numpy computation


    train_greyscale = rgb2gray(X_train).astype(np.float32)
    test_greyscale = rgb2gray(X_test).astype(np.float32)
    val_greyscale = rgb2gray(X_val).astype(np.float32)

    print("Training Set after greyscale conversion", train_greyscale.shape)
    print("Validation Set after greyscale conversion", val_greyscale.shape)
    print("Test Set after greyscale conversion", test_greyscale.shape)
    print('--'*10)
    print("Sanity check:")
    print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)

    del X_train, X_test, X_val

    #Ploting the Grayscale Image
    #Before Normalization

    plot_images(train_greyscale, y_train, 1, 10)

    #MEAN SUBTRACTION - Centering the data around the mean:
    #& NORMALIZATION - Diving the data by its standard deviation:


    train_mean = np.mean(train_greyscale, axis=0)

    train_std = np.std(train_greyscale, axis=0)

    # Subtract it equally from all splits
    train_greyscale_norm = (train_greyscale - train_mean) / train_std
    test_greyscale_norm = (test_greyscale - train_mean)  / train_std
    val_greyscale_norm = (val_greyscale - train_mean) / train_std







    plot_images(train_greyscale_norm, y_train, 1, 10)



    #One Hot Label Encoding
    #Apply One Hot Encoding to make label
    #suitable for CNN Classification

    from sklearn.preprocessing import OneHotEncoder
    
    # Fit the OneHotEncoder
    enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

    # Transform the label values to a one-hot-encoding scheme
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

    print("Training set", y_train.shape)
    print("Validation set", y_val.shape)
    print("Test set", y_test.shape)







    #Storing Data to Disk
    #Stored only the Grayscale Data 
    #not the RGB

    # Create file
    h5f = h5py.File('SVHN_grey(clean).h5', 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data=train_greyscale_norm)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=test_greyscale_norm)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('X_val', data=val_greyscale_norm)
    h5f.create_dataset('y_val', data=y_val)

    # Close the file
    h5f.close()





    #PRE-Processing --------------------------------------- END

    tbc=TensorBoardColab()

    tbtCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                            write_graph=True,
                            write_grads=True,
                            batch_size=64,
                            write_images=True)

    # Open the file as readonly
    h5f = h5py.File('SVHN_grey(clean).h5', 'r')

    print(list(h5f))
    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]

    # Close this file
    h5f.close()

    print('Training set', X_train.shape, y_train.shape)
    print('Validation set', X_val.shape, y_val.shape)
    print('Test set', X_test.shape, y_test.shape)

    #(Size, Width, Height, Channel (Colour))(Size, NumofClasses)



    def plot_model(model_info):

        # Create sub-plots
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        
        # Summarize history for accuracy
        axs[0].plot(range(1,len(model_info.history['acc'])+1),model_info.history['acc'])
        axs[0].plot(range(1,len(model_info.history['val_acc'])+1),model_info.history['val_acc'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_info.history['acc'])+1),len(model_info.history['acc'])/10)
        axs[0].legend(['train', 'val'], loc='bestmodel_info')
        # Summarize history for loss
        axs[1].plot(range(1,len(model_info.history['loss'])+1),model_info.history['loss'])
        axs[1].plot(range(1,len(model_info.history['val_loss'])+1),model_info.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_info.history['loss'])+1),len(model_info.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        
        # Show the plot
        plt.show()





















    # Hight and width of the images
    IMAGE_SIZE = 32
    # 1 Channel - Grey (NOT RBG)
    CHANNELS = 1
    # Number of epochs
    NUM_EPOCH = 1
    # learning rate
    LEARN_RATE = 1.0e-4


    def f1_score(y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2*p*r / (p+r+K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)
    
    
    
    
    def simple_cnn():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding = 'same', input_shape=(32, 32, 1)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))

        #The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
        
        
        model.add(Conv2D(32, (3, 3), padding='same', strides = 2))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # The next layer is the substitute of max pooling, we are taking a strided convolution layer to reduce the dimensionality of the image.
        
        
        model.add(Conv2D(64, (3, 3) ,padding='same'))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3),  padding = 'same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (1, 1),  padding='valid'))
        model.add(Activation('relu'))

        model.add(Conv2D(10, (1, 1), padding='valid'))
        model.add(Activation('relu'))

        
        
    
        
        model.add(GlobalAveragePooling2D())
        model.add(Activation('softmax'))
        
        
        
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', f1_score])
        
        return model
    

    def pure_cnn_model():
        
        model = Sequential()
        
        model.add(Conv2D(1, (3, 3), activation='relu', padding = 'same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))    
        model.add(Dropout(0.2))
        
        model.add(Conv2D(28, (3, 3), activation='relu', padding = 'same'))  
        model.add(Conv2D(28, (3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))    
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same', strides = 2))    
        model.add(Dropout(0.5))   
        
        model.add(Conv2D(28, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Conv2D(28, (1, 1),padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(10, (1, 1), padding='valid'))

        model.add(GlobalAveragePooling2D())
        
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
                optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
                metrics = ['accuracy', f1_score]) # Metrics to be evaluated by the model

        
        return model

    model0 = pure_cnn_model()
    model0.load_weights('weights.best.digits.pure_cnn.hdf5')



    print(model0.summary())
    cnn_pure_checkpointer = ModelCheckpoint(filepath='weights.best.digits.pure_cnn.hdf5', 
                                    verbose=2, save_best_only=True)



    #Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
    #This callback monitors a quantity and if no improvement is seen for a 
    #'patience' number of epochs, the learning rate is reduced.


    #Patience is the number of epochs before the model begins to update its parameters
    #number of epochs with no improvement after which learning rate will be reduced.


    #Factor - factor by which the learning rate will be reduced. new_lr = lr * factor
    #cnn_lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
    #                                     patience=40, verbose=2, factor=0.1)




    cnn_history = model0.fit(X_train, y_train, 
                                validation_data=(X_val, y_val), 
                                epochs=250, batch_size=32, verbose=2, 
                                callbacks=[TensorBoardColabCallback(tbc), cnn_pure_checkpointer])

    plot_model(cnn_history)

    scores=model0.evaluate(X_test, y_test, verbose=2)
    #serialize model to JSON


    print("Model 1's %s: %.2f%%" % (model0.metrics_names[1], scores[1]*100))





    """simple_Cnn_Model 

    Using batch size = 32
    epochs = 150
    """

    model1 = simple_cnn()
    print(model1.summary())
    model1.load_weights('weights.simple_cnn.hdf5')



    simple_cnn_checkpointer = ModelCheckpoint(filepath='weights.simple_cnn.hdf5', 
                                    verbose=2, save_best_only=True)



    #Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
    #This callback monitors a quantity and if no improvement is seen for a 
    #'patience' number of epochs, the learning rate is reduced.


    #Patience is the number of epochs before the model begins to update its parameters
    #number of epochs with no improvement after which learning rate will be reduced.


    #Factor - factor by which the learning rate will be reduced. new_lr = lr * factor
    #cnn_lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
    #                                     patience=40, verbose=2, factor=0.1)



    simple_cnn_history = model1.fit(X_train, y_train, 
                                validation_data=(X_val, y_val), 
                                epochs=NUM_EPOCH, batch_size=32, verbose=2, 
                                callbacks=[TensorBoardColabCallback(tbc), simple_cnn_checkpointer])

    scores=model1.evaluate(X_test, y_test, verbose=2)
    #serialize model to JSON


    print("Model 2's %s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))

    plot_model(simple_cnn_history)















    plot_model(cnn_history)



    import matplotlib.pyplot as plt
    from keras.preprocessing.image import ImageDataGenerator



    #Steps = number of samples per gradient update

    steps, epochs = len(X_train)/32, NUM_EPOCH


    data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images



    # fit the data augmentation
    data_generator.fit(X_train)

    dg_pure_cnn_history = model0.fit_generator(data_generator.flow(X_train, y_train, 
                                    batch_size=32),
                steps_per_epoch = steps, epochs = NUM_EPOCH,
                validation_data = (X_val, y_val), 
                callbacks=[TensorBoardColabCallback(tbc), cnn_checkpointer], verbose=2)

    plot_model(dg_pure_cnn_history)

    dg_simple_cnn_history = model1.fit_generator(data_generator.flow(X_train, y_train, 
                                    batch_size=32),
                steps_per_epoch = steps, epochs = NUM_EPOCH,
                validation_data = (X_val, y_val), 
                callbacks=[TensorBoardColabCallback(tbc)], verbose=2)

    plot_model(dg_simple_cnn_history)

    # configure batch size and retrieve one batch of images
    for X_batch, y_batch in data_generator.flow(X_train, y_train, batch_size=9):
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.imshow(X_batch[i].reshape(32, 32), cmap=plt.get_cmap('gray'))
        # show the plot
        plt.show()
        break

    scores = model1.evaluate(X_test, y_test, verbose=2)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    plot_model(dg_cnn_history)

    scores = model1.evaluate(X_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))






    print(set(list(cnn_history.history)))









    plt.figure(figsize=(14, 7))

    plt.plot(cnn_history.history['acc'], label = 'Base Model - Training Accuracy')
    plt.plot(cnn_history.history['val_acc'], label = 'Base Model Validation Accuracy')


    plt.plot(dg_simple_cnn_history.history['acc'], label = 'Augmented Model  -  Training Accuracy')
    plt.plot(dg_simple_cnn_history.history['val_acc'], label = 'Augmented Model - Validation Accuracy')



    plt.legend()
    plt.title('Accuracy - Training & Validation');

    plt.figure(figsize=(14, 7))

    plt.plot(cnn_history.history['f1_score'], label = ' Base Model - Training  F1 Score')
    plt.plot(cnn_history.history['val_f1_score'], label = 'Base Model - Validation F1 Score')


    plt.plot(dg_cnn_history.history['f1_score'], label = 'Augmented Model - Training F1 Score')
    plt.plot(dg_cnn_history.history['val_f1_score'], label = 'Augmented Model - Validation F1 Score')


    plt.legend()
    plt.title('F1 Score - Training & Validation');

    plt.figure(figsize=(14, 7))

    plt.plot(dg_cnn_history.history['val_loss'], label = 'Augmented Images - Validation Loss')
    plt.plot(dg_cnn_history.history['loss'], label = 'Augmented Images - Training Loss')


    plt.plot(cnn_history.history['val_loss'], label = 'Non-Augmented Images - Validation Loss')
    plt.plot(cnn_history.history['loss'], label = 'Non-Augmented Images - Training Loss')

    plt.legend()
    plt.title('Learning Rate & Loss - Model using Augmented Images vs.  Without augmented Images');


    path = 'Models'
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)



    model0.save("Models/model-1.h5")
    print("Saved model-1 to disk")

    model1.save("Models/model-2.h5")
    print("Saved model-2 to disk")



    # image folder
    folder_path = '/path/to/folder/'
    # path to model
    model_path = '/path/to/saved/model.h5'
    # dimensions of images
    img_width, img_height = 320, 32

    # load the trained model
    def test(folder_path, model, images=Null):
    # load all images into a list
        images = []
        for image in os.listdir(folder_path):
            image = image.load_img(image, target_size=(image_width, image_height))
            image = image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            images.append(image)

    # stack up images list to pass for prediction
    images = np.vstack(images)
    classes = model.predict_classes(images, batch_size=10)
    print(classes)

if __name__ == "__main__":
    main()












