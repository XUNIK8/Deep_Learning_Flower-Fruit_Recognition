import os
path = 'E:/Dataset/barbeton daisy'   #文件夹位置
files = os.listdir(path)
for i, file in enumerate(files):
    NewName = os.path.join(path, "barbeton daisy"+str(i+1)+'.jpg')   #新名字：种类+序号
    OldName = os.path.join(path, file)
    os.rename(OldName, NewName)