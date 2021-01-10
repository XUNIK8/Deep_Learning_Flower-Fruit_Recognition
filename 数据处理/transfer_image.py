
from PIL import Image
import cv2 as cv
import os

im_path="/home/group6/deep_learning/data/images/"
train_path="/home/group6/deep_learning/experiment/zcl/images"
val_path="/home/group6/deep_learning/experiment/zcl/val/" #验证集路径
test_path="/home/group6/deep_learning/experiment/zcl/test/" # 测试集路径

def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)
        os.remove(PngPath)



def transfer(file_name,path):
    img_path = os.path.join(path, file_name)
    img_files = os.listdir(img_path)

    for filename in img_files:
        newname=filename
        newname = newname.split(".")
        if newname[-1] in ["JPEG","jpeg"]:
            newname[-1] = "jpg"
            newname = str.join(".", newname)# 这里要用str.join
            filename = os.path.join(img_path,filename)
            newname = os.path.join(img_path,newname)
            os.rename(filename, newname)
            print(newname, "updated successfully")
        elif newname[-1] in ["jpg","JPG"]:
            continue
        elif newname[-1] in ["png","PNG"]:
            PNG_JPG(os.path.join(img_path,filename))

for files_1 in os.listdir(im_path):
    transfer(files_1, im_path)

# for files_2 in os.listdir(test_path):
#     transfer(files_2, test_path)
#
# for files_3 in os.listdir(val_path):
#     transfer(files_3, val_path)







