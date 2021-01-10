import numpy as np
import os
import shutil
# 图片文件夹的地址，请务必复制一份到自己的文件夹下进行操作
path="/home/group6/deep_learning/experiment/zcl/images"
files = os.listdir(path)
#只需要把zcl改了就行
val_path="/home/group6/deep_learning/experiment/zcl/val/" #验证集路径
test_path="/home/group6/deep_learning/experiment/zcl/test/" # 测试集路径
os.mkdir(val_path)
os.mkdir(test_path)


def split_val_test(file_name, val_rate, test_rate):
    #随机数可以改一下 不同随机数就是不同切分方法
    np.random.seed(10)
    path = "/home/group6/deep_learning/experiment/zcl/images" #与上面path相同
    img_path = os.path.join(path, file_name)
    img_files = os.listdir(img_path)

    np.random.shuffle(img_files)
    num = len(img_files)
    val_num = int(num * val_rate)
    test_num = int(num * test_rate)
    val_file = img_files[:val_num]
    test_file = img_files[val_num:val_num + test_num]
    #### split val_file
    os.mkdir(os.path.join(val_path, file_name))
    for img in val_file:
        val_img = os.path.join(img_path, img)
        shutil.move(val_img, os.path.join(val_path, file_name))
    #### split test file
    os.mkdir(os.path.join(test_path, file_name))
    for img_2 in test_file:
        test_img = os.path.join(img_path, img_2)
        shutil.move(test_img, os.path.join(test_path, file_name))

for file_name in files:
    split_val_test(file_name,0.1,0.1)
