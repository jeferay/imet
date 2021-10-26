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
#from albumentations import HorizontalFlip, OneOf, RandomBrightness, RandomContrast, ShiftScaleRotate, IAAAdditiveGaussianNoise
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.utils.data as data
from pathlib import Path
import math
import random


#定义RandomErasing类,实现随机擦除
class RandomErasing(object):
	'''
	probability:执行擦除操作的概率（体现随机性）
	sl: 擦除面积的下限
	sh: 擦除面积的上限
	r1: 擦除区域的长宽比界限，取区间（rl,1/rl）
	mean: erasing value
	'''
	
	def __init__(self, probability=0.3, sl=0.02, sh=0.3, r1=0.3, mean=[0.485, 0.456, 0.406]):
        #这里mean的参数设置可调整
		self.probability = probability
		self.mean = mean
		self.sl = sl
		self.sh = sh
		self.r1 = r1
	
	def __call__(self, img):
		#按一定的概率决定是否执行Erasing操作
		if np.random.uniform(0, 1) > self.probability:
			return img
		

		area = img.size()[1] * img.size()[2]
		for attempt in range(100):#这里的100次可调整
			target_area = np.random.uniform(self.sl, self.sh) * area#目标擦除区域面积
			aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)#目标擦除区域宽高比
			
            #计算目标擦除区域的宽和高
			h = int(round(math.sqrt(target_area * aspect_ratio)))
			w = int(round(math.sqrt(target_area / aspect_ratio)))
			
			if w < img.size()[2] and h < img.size()[1]:
                #随机选取擦除区域：通过随机选出擦除区域左上角点的坐标得到
				x1 = np.random.randint(0, img.size()[1] - h)
				y1 = np.random.randint(0, img.size()[2] - w)
                
				if img.size()[0] == 3:#RGB图像用这个
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
					img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
					img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
				else:#非RGB
					img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
				return img
		
		return img

class RandomResizedCropV2(transforms.RandomResizedCrop):

    @staticmethod
    def get_params(img, scale, ratio):

        # ...

        # fallback
        w = min(img.size[0], img.size[1])
        i = random.randint(0, img.size[1] - w)
        j = random.randint(0, img.size[0] - w)

        return i, j, w, w

#数据预处理
trans_train= transforms.Compose([
    #HorizontalFlip(p=0.5),
    #OneOf([
    #        RandomBrightness(0.1, p=1),
    #        RandomContrast(0.1, p=1),
    #], p=0.3),
    #ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=0.3),
    #IAAAdditiveGaussianNoise(p=0.3),
    RandomResizedCropV2((288,288), scale=(0.7, 1.0), ratio=(4/5, 5/4)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]),
    RandomErasing()
    
])
trans_valid= transforms.Compose([
    RandomResizedCropV2((288,288), scale=(0.7, 1.0), ratio=(4/5, 5/4)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])
])


#定义第一层dataset类
class MyDataset_stack1(Dataset):   #MyDataset继承了Dataset类，该类并不保存图片，只是在读取时选择对应图片路径并返回图片
    def __init__(self, df_data, mode, data_dir= './',transform=None):
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
    def __getitem__(self, index):#获取元素时候再读入图片
        if(self.mode == 'train'):
            img_name = self.df[index][0]   
        else:
            img_name = self.df[index]
        img_path = os.path.join(self.data_dir,img_name+'.png') #有关于系统路径问题
        image = Image.open(img_path).convert('RGB')#转化为RGB图像
        image = self.transform(image)#用对应的transform预处理
        if(self.mode == 'train'):
            label = self.df[index][2]
            label_tensor = np.zeros((1, 3474))
            for i in label:
                label_tensor[0, int(i)] = 1          
            label_tensor =label_tensor.flatten()   #变为一维
            label_tensor =torch.from_numpy(label_tensor).float()
            return image,label_tensor
        else: 
            return image

#定义第二层dataset类
class MyDataset_stack2(Dataset):
    def __init__(self, mode, data_dir,length,id_data):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.length = length 
        self.id_data = id_data
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        if self.mode =='valid':index = 5 * 23680 +index#最后一折数据
        vector = pd.read_csv(self.data_dir,index_col=index)
        vector =torch.Tensor(vector).reshape(3,18,193)#重新整理规模
        label = self.id_data[index][2]#取得labels数据
        label_tensor = np.zeros((1, 3474))
        for i in label:
            label_tensor[0, int(i)] = 1          
        label_tensor =label_tensor.flatten()   #变为一维
        label_tensor =torch.from_numpy(label_tensor).float()
        return vector,label_tensor


