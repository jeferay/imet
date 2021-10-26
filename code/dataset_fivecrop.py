"""
该文件用来定义加载和预处理数据的类和函数
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.utils.data as data
from pathlib import Path
import math

num_class = 3474


# 定义RandomErasing类,实现随机擦除
class RandomErasing(object):
    '''
    probability:执行擦除操作的概率（体现随机性）
    sl: 擦除面积的下限
    sh: 擦除面积的上限
    r1: 擦除区域的长宽比界限，取区间（rl,1/rl）
    mean: erasing value
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        # 这里mean的参数设置可调整
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # 按一定的概率决定是否执行Erasing操作
        if np.random.uniform(0, 1) > self.probability:
            return img

        area = img.size()[1] * img.size()[2]
        for attempt in range(100):  # 这里的100次可调整
            target_area = np.random.uniform(self.sl, self.sh) * area  # 目标擦除区域面积
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)  # 目标擦除区域宽高比

            # 计算目标擦除区域的宽和高
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                # 随机选取擦除区域：通过随机选出擦除区域左上角点的坐标得到
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)

                if img.size()[0] == 3:  # RGB图像用这个
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:  # 非RGB
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# 数据预处理
trans_train = transforms.Compose([
    transforms.RandomResizedCrop(288),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]),
    #RandomErasing()  # 使用默认参数，随机擦除
])
trans_valid1 = transforms.Compose([
    transforms.Resize(320),
    transforms.FiveCrop(288),  # 此时返回五张图片的tuple
    # transforms.ToTensor(),
    transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops]))
    # 将每张照片转化为tensor后堆叠，returns a 4D tensor5*channel*height*width
])

trans_valid2 = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(288),  # 此时返回五张图片的tuple
    transforms.ToTensor(),
    #transforms.Lambda(lambda crops: torch.stack([(transforms.ToTensor()(crop)) for crop in crops]))
    # 将每张照片转化为tensor后堆叠，returns a 4D tensor5*channel*height*width
])

trans_valid3 = transforms.Compose([
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])


# 定义dataset类
class MyDataset(Dataset):  # MyDataset继承了Dataset类，该类并不保存图片，只是在读取时选择对应图片路径并返回图片
    def __init__(self, df_data, mode, data_dir='./', transform=None):
        """
        args:
        id_data：panda读出的csv数据,已经转化为list of (id,attribute_ids(为string类型),list of (attribute_ids))
        mode:'train'或者'test'
        data_dir:表明图片文件夹所在路径
        transform:对图片的transform操作
        """
        super().__init__()
        self.df = df_data
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):  # 获取元素时候再读入图片
        if (self.mode == 'train'):
            img_name = self.df[index][0]
        else:
            img_name = self.df[index]
        img_path = os.path.join(self.data_dir, img_name + '.png')  # 有关于系统路径问题
        image = Image.open(img_path).convert('RGB')  # 转化为RGB图像
        if self.transform is 'trans_train':
            image = trans_train(image)  # 训练集用trans_train
        else:
            image_five = trans_valid1(image)  # 测试集和验证集用trans_valid
            for i in range(5): image_five[i] = trans_valid3(image_five[i])
            image_center = trans_valid2(image)#five:5*3*height*width
            image_center=trans_valid3(image_center)#center:3*height*width
            image_center=torch.unsqueeze(image_center,0)
            #image = image_five
            image = torch.cat((image_center,image_five),0)#合并为6*3*height*width

        if (self.mode == 'train'):
            label = self.df[index][2]
            label_tensor = np.zeros((1, 3474))
            for i in label:
                label_tensor[0, int(i)] = 1
            label_tensor = label_tensor.flatten()  # 变为一维
            label_tensor = torch.from_numpy(label_tensor).float()
            return image, label_tensor
        else:
            return image
