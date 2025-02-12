# %%
#Label 0-T2CEP 1-RR 2-HYB 3-GDOR 4-EB 5-DSCT
import os
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

# 读取文件夹下的每个文件
path = "T2CEP"
label = '0-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)
    
    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

# 删除前一轮内存
del cwtmatr, freqs, data, file_path, file, files, path, label
# 读取文件夹下的每个文件
path = "RR"
label = '1-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

# 删除前一轮内存
del cwtmatr, freqs, data, file_path, file, files, path, label
# 读取文件夹下的每个文件
path = "HYB"
label = '2-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

# 删除前一轮内存
del cwtmatr, freqs, data, file_path, file, files, path, label
# 读取文件夹下的每个文件
path = "GDOR"
label = '3-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

# 删除前一轮内存
del cwtmatr, freqs, data, file_path, file, files, path, label
# 读取文件夹下的每个文件
path = "EB"
label = '4-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

# 删除前一轮内存
del cwtmatr, freqs, data, file_path, file, files, path, label
# 读取文件夹下的每个文件
path = "DSCT"
label = '5-CWT'
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path)

    # 提取时序数据
    dt = 0.02
    data = data.iloc[:, 1]
    # 将data进行标准化处理
    data = (data - data.min()) / (data.max() - data.min())
    # 进行CWT变换，并保存图像
    # 水平轴是时间，垂直轴是频率
    cwtmatr, freqs = pywt.cwt(data, np.arange(1, 128), 'morl')
    cwtmatr = np.abs(cwtmatr)
    plt.imshow(cwtmatr, extent=[0, len(data) * dt, freqs[-1], freqs[0]], cmap='PRGn', aspect='auto',
        vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.axis('off')
    plt.savefig(label+'/' + file + '.jpg')
    plt.close()

