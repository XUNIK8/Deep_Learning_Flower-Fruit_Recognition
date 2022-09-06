# 详细介绍
- 知乎说明文档：https://zhuanlan.zhihu.com/p/343259867
- 介绍视频：https://www.bilibili.com/video/BV1rK411M7Vn?spm_id_from=333.999.0.0&vd_source=fbf5a86c948d55005a3ddf0f63195916

# 文件说明

## 数据处理：

- rename.py - 文件重命名

- split_val_test.py - 划分数据集

- transfer_image.py - 转换图片格式

- get_data.py - 爬取图片数据集

## 模型训练

- AlexNet.py -  AlexNet 网络训练

- GoogleNet.py - GoogleNet网络训练

- MobileNet.py -  MobileNet和残差网络训练

- final_model.h5 - 最终训练模型

## 部署

- recognition_1.py - 初始部署版本

- recognition_2.py - 优化部署版本

- transfer_lite.py -  h5转tflite

- model.h5 - 最终部署模型
