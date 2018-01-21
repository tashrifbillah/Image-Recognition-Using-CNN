# Training the CNN to extract features
import os
import pickle  # loading images
import random
import time

import numpy as np
import scipy as sp  # loading images
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Convolution2D
from keras.layers import merge
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from scipy import misc  # loading images
from sklearn.metrics import confusion_matrix


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    # Parellel GPU execution had some problems
    # Please feel free to contribute
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

start_time = time.time()

def load_selected_data(l):

    y = pickle.load(open("data.p","rb"))
    x = []
    for i in range(len(y)):
        
        # There are 10 subimages. You can load all of them if you have a 
        # powerful GPU and adequate memory capacity. But for first hand
        # trial, you can just load any one index out of 0-9
        if len(y[i]) == 1 and (i%10 in (1,4,9)) and y[i][0] == l:
            x.append(sp.misc.imread(os.path.join("out", "%i.png" % i)))


    Nl,_,_,_= np.shape(x)
    y= l*np.ones((Nl,1),dtype= float)

    return (x,y)


def load_sample_data():

    trainx= [ ]
    trainy= [ ]
    testx= [ ]
    testy= [ ]
    for l in range(20):

        xl, yl = load_selected_data(l)

        N= len(yl)
        eta= 0.8 # Train ratio

        names= np.array(random.sample(set(np.arange(N)), int(N*eta)))
        Train_index = np.where(np.in1d(np.arange(N),names))
        Test_index = np.setdiff1d(np.arange(N), Train_index)

        xl = np.array(xl)
        yl = np.array(yl)

        trainx.extend([xl[i] for i in Train_index[0]])
        trainy.extend([yl[i] for i in Train_index[0]])

        testx.extend([xl[i] for i in Test_index])
        testy.extend([yl[i] for i in Test_index])

    trainx = np.array(trainx)
    trainy = np.array(trainy)
    testx = np.array(testx)
    testy = np.array(testy)

    x = np.concatenate((trainx, testx))
    mean = np.ndarray.astype(np.mean(x,(0,1,2)),dtype=np.int8)
    trainx = trainx-mean
    testx = testx-mean

    trainx = trainx.reshape(trainx.shape[0], trainx.shape[1], trainx.shape[2], 3)
    testx = testx.reshape(testx.shape[0], testx.shape[1], testx.shape[2], 3)

    return trainx, trainy, testx, testy


def train_cnn(x,y):

    model = Sequential()
    # Layer 1

    model.add(Convolution2D(96, (7, 7), strides= 2, activation='relu', input_shape=(225,225,3)))
    model.add(MaxPooling2D(pool_size=(3, 3),strides= 2))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Convolution2D(256, (5, 5), strides= 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization())

    # Layer 3
    model.add(Convolution2D(384, (3, 3), activation='relu'))

    # Layer 4
    model.add(Convolution2D(384, (3, 3), activation='relu'))

    # Layer 5
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layer 6
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 7
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(20, activation='softmax'))
    
    # Parellel GPU execution had some problems
    # Please feel free to contribute
    # model = make_parallel(model, 4)
    
    sgd = optimizers.SGD(lr=0.01, momentum=0.9)
    model = multi_gpu_model(model, gpus=4)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history= model.fit(x,y, batch_size=32, epochs=25)

    # # Plot the Loss Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(history.history['loss'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    # plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.title('Loss Curves', fontsize=16)
    # plt.grid(True)
    # plt.show(block=False)
    #
    #
    # # Plot the Accuracy Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(history.history['acc'], 'r', linewidth=3.0)
    # plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    # plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Accuracy Curves', fontsize=16)
    # plt.grid(True)
    # plt.show(block=False)

    return model


trainx, trainy, testx, testy = load_sample_data()

trainy= to_categorical(trainy,20)
model= train_cnn(trainx,trainy)

# model.save('my_model.h5')

# 4. model evaluation
prob = model.predict(testx)
temp= np.argmax(prob,axis=1).reshape(len(prob),1)

testy = np.reshape(testy,(len(testy),))
temp = np.reshape(temp,(len(temp),))

# 5. error calculation
acc= np.sum(temp==testy)/len(testy)
print("Average accuracy over all classes: %f" % acc)

print("Confusion matrix")
print(confusion_matrix(temp,testy))
for i in range(20):
    ind1= np.where(testy==i)[0]
    acc= len(np.where(temp[ind1]==i)[0])/ len(ind1)
    print("Accuracy of class %d: %f" % (i, acc))

print("--- Execution time: %s seconds ---" % (time.time() - start_time))

print("Wait")
