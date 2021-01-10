import os
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.optimizers import Adam, SGD,RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

import time
import numpy as np
from PIL import Image
from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.utils import multi_gpu_model

# 将 `model` 复制到 8 个 GPU 上。
# 假定你的机器有 8 个可用的 GPU。
import tensorflow as tf
import random
import pathlib



from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Dropout,Flatten,BatchNormalization,add,Activation,GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
from tensorflow.keras import regularizers

train_path="/home/group6/deep_learning/experiment/zcl/images"
val_path='/home/group6/deep_learning/experiment/zcl/val'
test_path='/home/group6/deep_learning/experiment/zcl/test'


def get_img_num(path):
    file_list=os.listdir(path)
    img_num=0
    for files in file_list:
        sun_path=os.path.join(path,files)
        sub_num=len(os.listdir(sun_path))
        img_num=img_num+sub_num
    return img_num

input_shape=[120,120]
batch_size=512
val_size=300
total_epoch=300
train_img_num=get_img_num(train_path)
#val_img_num=get_img_num(val_path)
#test_img_num=get_img_num(test_path)
np.random.seed(10)

def load_preprosess_image(input_path):
    image = tf.io.read_file(input_path)  # 读取的是二进制格式 需要进行解码
    image = tf.image.decode_jpeg(image, channels=3)  # 解码 是通道数为3
    image = tf.image.resize(image,input_shape)  # 统一图片大小

    image = image / 255.0  # 归一化
    return image


def get_image(path, batch_size, input_size, NUM, kinds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data_path = pathlib.Path(path)
    all_image_paths = list(data_path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]  # 所有图片路径的列表
    image_count = len(all_image_paths)
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
    # one hot
    all_image_labels = tf.one_hot(all_image_labels, depth=NUM)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = image_label_ds.shuffle(buffer_size=image_count)
    if kinds=="train":
        ds = ds.repeat()
        ds = ds.batch(batch_size)
        # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    else:
       # ds = ds.repeat()
        ds = ds.batch(val_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
def residual_block(x, num,filters,size):
    shortcut = x
    x = Conv2D(filters, (size, size), strides=1, padding='same', name='resi_conv_%d_1' % num)(x)
    x = BatchNormalization(name='resi_normal_%d_1' % num)(x)
    x = Activation('relu', name='resi_relu_%d_1' % num)(x)
    x = Conv2D(filters, (size, size), strides=1, padding='same', name='resi_conv_%d_2' % num)(x)
    x = BatchNormalization(name='resi_normal_%d_2' % num)(x)
    m = add([x, shortcut], name='resi_add_%d' % num)
    return m

def fruit_model(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(16, (7, 7), activation='relu', name="Conv2D_1")(X_input)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(X)

    X = Conv2D(16, (4, 4), activation='relu', name="Conv2D_2", kernel_regularizer=regularizers.l1(0.01))(X)
    X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(X)
    X = Conv2D(32, (4, 4), activation='relu', name="Conv2D_3", kernel_regularizer=regularizers.l1(0.01))(X)
    X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(X)
    # X=MaxPooling2D(pool_size=(4,4),strides=1, padding='valid')(X)
    # X=Conv2D(32, (3, 3),activation='relu',name="Conv2D_9")(X)
    # X=Conv2D(16, (5, 5),activation='relu',name="Conv2D_12")(X)
    X=residual_block(X,1,32,3)
    X=residual_block(X,2,32,3)
   # X = Conv2D(32, (4, 4), activation='relu', name="Conv2D_10")(X)
   # X=residual_block(X,3,32,3)
   # X=residual_block(X,4,32,3)
   # X=residual_block(X,5,32,4)
   # X = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(X)
   # X = Conv2D(32, (5, 5), activation='relu', name="Conv2D_3")(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(X)

   # X = Conv2D(32, (4, 4), activation='relu', name="Conv2D_13", kernel_regularizer=regularizers.l2(0.01))(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(X)

   # X = Conv2D(64, (4, 4), activation='relu', name="Conv2D_8")(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(X)

   # X = Conv2D(64, (3, 3), activation='relu', name="Conv2D_10", kernel_regularizer=regularizers.l2(0.01))(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(X)

 #   X = Conv2D(128, (3, 3), activation='relu', name="Conv2D_4")(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(X)

  #  X = Conv2D(128, (3, 3), activation='tanh', name="Conv2D_9")(X)
   # X = BatchNormalization(momentum=0.99, epsilon=0.001)(X)
   # X = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(X)

    #     X=Conv2D(10, (3, 3),activation='relu',name="Conv2D_5")(X)
    #     X=BatchNormalization( momentum=0.99, epsilon=0.001)(X)
    #     X=MaxPooling2D(pool_size=(16,16),strides=1, padding='valid')(X)
    
    X = Flatten()(X)
   # X=GlobalAveragePooling2D(X)
   # X = Dense(50, activation='softmax', name='FC_1_softmax')(X)
    X = Dropout(0.5)(X)
    X = Dense(26, activation='softmax', name='FC_2_softmax')(X)
    model = Model(inputs=X_input, outputs=X, name='fruit_model')
    #opt = SGD(lr=0.01, momentum=0.0, decay=0.000001, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model

def mobile():
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
    mobile_net.trainable = True
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(26, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model


def train(total_epoch, train_path, input_shape):
    # datagen = ImageDataGenerator(rescale=1. / 255,validation_split = 0.1,width_shift_range = 0.15,
    # height_shift_range = 0.15,
    # shear_range = 0.15,rotation_range = 25)
    save_dir = "/home/group6/deep_learning/experiment/zcl/model_2"
    # train_generator = datagen.flow_from_directory(
    #     train_path,
    #     target_size=input_shape,
    #     batch_size=batch_size,
    #     seed=1,subset='training',
    #     class_mode='categorical')
    #
    #
    # validation_generator = datagen.flow_from_directory(
    #     train_path,
    #     target_size=input_shape,
    #     batch_size=300,
    #     seed=15,
    #     class_mode='categorical',
    #     subset='validation')

    train_set=get_image(train_path, batch_size, input_shape, 26, "train")
    val_set=get_image(val_path, val_size, input_shape, 26, "val")

    #def change_range(image, label):
     #   return 2 * image - 1, label

   # train_set = train_set.map(change_range)
   # val_set=val_set.map(change_range)




    our_model = fruit_model((input_shape[0], input_shape[1], 3))
    #our_model=mobile()
    save_fname = os.path.join(save_dir, 'model_at_{epoch:02d}.h5' )  # 将时间和epoch id存到模型名称中
    callback = [ModelCheckpoint(filepath=save_fname, monitor='val_acc', save_best_only=False, save_freq=200, mode='max')]
    print(our_model.summary())
    print("开始训练")
    History = our_model.fit(
        train_set,
        steps_per_epoch=train_img_num // batch_size,
        epochs=total_epoch,
        validation_data=val_set,
        validation_steps=10,
	callbacks=callback
    )
    print("训练完成")
    our_model.save("/home/group6/deep_learning/experiment/zcl/model_1/final_model.h5")
    print("保存成功")
    #####可视化
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('/home/group6/deep_learning/experiment/zcl/model_2/pic/acc.png')
    plt.clf()
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('/home/group6/deep_learning/experiment/zcl/model_2/pic/loss.png')

if __name__ == "__main__":
    train(total_epoch, train_path, input_shape)
