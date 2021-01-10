# coding=gbk
#import tensorflow

import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
import keras
import time
from PIL import Image,ImageFile
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ImageFile.LOAD_TRUNCATED_IMAGES = True

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig("ac-loss.png")
      #  plt.show()

def imgToMat_RGB(img):
    img = Image.open(img)
    img = img.convert("RGB")
    #data = img.getdata()
    img = img.resize((227,227))
    mat = np.array(img)/255
    mat = mat.astype(np.float64)
    return mat
     

def get_train_data(data_folder_path):                        
    dirs = os.listdir(data_folder_path)
    data = []
    labels = []
    for dir_name in dirs:
        i = 0;
        label = int(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_name = os.listdir(subject_dir_path)
        for image_name in subject_images_name:
            i+=1
            if(i>=400):
               break
            image_path = subject_dir_path + "/" + image_name
            image = imgToMat_RGB(image_path)
            data.append(image)
            labels.append(label)
    #print(labels)
    return np.array(data),np.array(labels)

def shuffle_data(dir_name):
    X,Y = get_train_data(dir_name)
   # print(Y,"1")
    Y = keras.utils.to_categorical(Y)
   # print(Y,"2")
    index = [i for i in range(len(Y))] # test_data为测试数据
    np.random.seed(1)
    np.random.shuffle(index) # 打乱索引
    train_data = X[index]
    train_labels = Y[index]
    print(train_labels)
    return train_data,train_labels

def define_model():
    model = Sequential()
    model.add(Conv2D(filters=96,
                     kernel_size=(11,11),
                     strides=(4,4),
                     padding='same',
                     input_shape=(227,227,3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2),
                           padding='valid'))
    
    model.add(Conv2D(filters=256,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2),
                           padding='valid'))
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu'))
   
    model.add(MaxPooling2D(pool_size=(3,3),
                           strides=(2,2),
                           padding='valid')) 

    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
  
    model.add(Dense(25,activation='softmax'))
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def train_model():                               
    history = LossHistory()
    start_time = time.time()
    model = define_model()
#    model = load_model("model1.h5")
    train_data,train_labels = shuffle_data("data/train")
    X_test,Y_test = shuffle_data("data/test")
    #print(train_data)
    model.fit(train_data,train_labels,epochs=1,validation_data=(X_test, Y_test),
            callbacks=[history])
    model.save('model.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])    




    end_time = time.time()
    run_time = (end_time-start_time)
    print(run_time)
#    for i in range(100):                               
#        start_time = time.time()
    #model = define_model()
#        model = load_model("model"+str(i)+".h5")
   # train_data,train_labels = shuffle_data()
    #print(train_data)
#        model.fit(train_data,train_labels,epochs=100)
#        model.save('model'+str(i+1)+'.h5')
       
#        end_time = time.time()
#        run_time = (end_time-start_time)
#        print(run_time)
    history.loss_plot('epoch')
train_model()
