#---------
#程序运行
#---------
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

from stack1 import Stack1
from stack2 import Stack2
import models

#一些全局数据
num_classes=3474
num_epoches=40
batch_size = 32
learning_rate = 2e-4
valid_fold = 0
threshold = [0.1]*5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name='se_resnet50'
data_dir='/home/wzr/imet/'
weight_dacay = 1e-7



stack1=Stack1(num_classes=num_classes,lr=learning_rate,device=device,num_epoches=num_epoches,batch_size=batch_size,threshold=threshold,model_name=model_name,data_dir=data_dir,valid_fold=valid_fold,weight_dacay = weight_dacay)
