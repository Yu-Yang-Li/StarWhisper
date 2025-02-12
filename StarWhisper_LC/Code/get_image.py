# %%
# Label 0-T2CEP 1-RR 2-HYB 3-GDOR 4-EB 5-DSCT
import pandas as pd
import numpy as np
import cv2
import os

# 获取图片
folder_path = 'T2CEP'
label = '0'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)

# 删除内存中的数据
del data, time, values, ratio, image

# 获取图片
folder_path = 'RR'
label = '1'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)

# 删除内存中的数据
del data, time, values, ratio, image

# 获取图片
folder_path = 'HYB'
label = '2'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)

# 删除内存中的数据
del data, time, values, ratio, image

# 获取图片
folder_path = 'GDOR'
label = '3'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)

# 删除内存中的数据
del data, time, values, ratio, image

# 获取图片
folder_path = 'EB'
label = '4'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)

# 删除内存中的数据
del data, time, values, ratio, image

# 获取图片
folder_path = 'DSCT'
label = '5'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    # 读取文件中的数据
    data = pd.read_csv(file_path)
    # 获取时间和数据
    time = data.iloc[:, 0]
    values = data.iloc[:, 1]
    # 将values进行标准化处理
    values = (values - values.min()) / (values.max() - values.min())
     # 求比值
    ratio = time / values
    # 创建三通道图像
    image = np.stack([time, values, ratio], axis=-1)
    # 使用 OpenCV 库将图像保存到对应标签文件夹中
    cv2.imwrite(label+'/' + file_name + '.jpg', image)