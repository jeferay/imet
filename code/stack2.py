import os
import time
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

from dataset import MyDataset_stack1,MyDataset_stack2,trans_train,trans_valid
#import models
import models

class Stack2(nn.Module):
    def __init__(self,num_classes,lr,device,num_epoches,batch_size,threshold,model_name,data_dir,valid_fold,weight_dacay):
        super(Stack2,self).__init__()
        
        #超参数设定
        self.num_classes=num_classes
        self.lr=lr
        self.num_epoches=num_epoches
        self.batch_size=batch_size
        self.threshold=threshold
        self.model_name=model_name
        self.data_dir=data_dir
        self.device=device
        self.bound=[0,100,781,786,2706,3474]
        self.valid_fold = valid_fold
        self.weight_dacay = weight_dacay
        
        if model_name=='resnet50':
            self.model = models.resnet50(num_classes=3474).to(self.device)
        if model_name=='se_resnext50':
            self.model = models.se_resnext50(num_classes=3474).to(self.device)
        if model_name=='airnext50':
            self.model = models.airnext50(num_classes=3474).to(self.device)
        if model_name=='resnet101':
            self.model = models.resnet101(num_classes=3474).to(self.device)
        if model_name=='se_resnext101':
            self.model = models.se_resnext101(num_classes=3474).to(self.device)
        
        self.criterion=nn.BCEWithLogitsLoss(reduction='mean')#对最后一层(线形层之后的结果)每个神经元做逻辑斯蒂回归
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=weight_dacay)#第一个阶梯的lr

        self.dataset_train,self.dataset_valid,self.dataset_test=self.get_dataset()
        self.parameter_path=data_dir+model_name +'.pth'#生成参数的保存路径

        #得到dataset类
    def get_dataset(self):
        #这一部分只是为了取得labels数据
        train_id_data= pd.read_csv(self.data_dir+"train.csv")
        y =train_id_data.attribute_ids.map(lambda x: x.split()).values   #将attribute每个值进行划分，因为一个值包含多个标签
        train_id_data['y']= y  #:增加纵向列为label(int)的lsit
        train_id_data = train_id_data.values#用values可以取到ndarray of (id,attribute_ids(为string类型),list of (attribute_ids))
        train_id_data,valid_id_data = train_id_data[0:23680 * 5],train_id_data[25680 * 5:]#分折完成，关键是保存labels
        
        #stack2的test输出则是从图片开始的，这部分完全等价于stack1
        test_id_data= pd.read_csv(self.data_dir+"sample_submission.csv")
        test_id_data= test_id_data['id']
        test_id_data = test_id_data.values#用values可以取到ndarray of id
        test_path= self.data_dir+'test/'

        dataset_train = MyDataset_stack2(mode='train',data_dir = self.data_dir + 'train_stack.csv',length = 118400,id_data = train_id_data)
        dataset_valid = MyDataset_stack2(mode='valid',data_dir = self.data_dir + 'valid_stack.csv',length = 23719,id_data = valid_id_data)
        dataset_test = MyDataset_stack1(df_data=test_id_data, mode='test', data_dir=test_path,transform=trans_valid)#采用第一层的读入方式
        return dataset_train, dataset_valid, dataset_test
        
    def train(self,pre_trained = False):
        loader_train=DataLoader(dataset=self.dataset_train,batch_size=self.batch_size,shuffle=True,num_workers=8)
        loader_valid=DataLoader(dataset=self.dataset_valid,batch_size=self.batch_size,shuffle=False,num_workers=8)
        min_val_loss = np.inf#记录最小loss
        sym,times=0,0#用来控制学习率的衰减,记录不减loss数和衰减次数
        if pre_trained:self.model.load_state_dict(torch.load(self.parameter_path))
        bestmodel=self.model.state_dict()#用来记录模型参数
        start_time=time.time()
        print('start training')             
        
        for epoch in range(self.num_epoches):
            start=time.time()
            avg_loss = 0
            #训练
            for iteration,(vectors,labels) in enumerate(loader_train):
                vectors=vectors.to(self.device)#shape:batch*in_channels*height*width
                labels=labels.to(self.device)#shape:batch*num_classes
                self.optimizer.zero_grad()
                outputs=self.model.forward(vectors)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                #if iteration%200==0:print('epoch', epoch,' iteration',iteration,'finished ,taking',(time.time()-start)/60,'min')
            
            #在验证集上计算loss和f2
            current_F2 = self.eval(pre_trained=False)#计算f2，已在内部断开梯度累计
            with torch.no_grad():
                for i,(vectors, labels) in enumerate(loader_valid):
                    vectors = vectors.to(self.device)#shape:batch*in_channels*height*width
                    labels = labels.to(self.device)#shape:batch*num_classes
                    outputs = self.model.forward(vectors)#shape:batch*num_calsses
                    loss = self.criterion(outputs, labels)
                    avg_loss += loss.item() /len(loader_valid)
            
            #通过patience实现衰减机制
            if min_val_loss > avg_loss:
                bestmodel = self.model.state_dict()#以loss为标准保存最好模型
                min_val_loss = avg_loss
                sym = 0
            else:#不减
                sym+=1
                if (sym>=2 and times<3):
                    newlr=0
                    sym=0
                    times+=1
                    for p in self.optimizer.param_groups:
                        p['lr'] *= 1/7
                        newlr=p['lr']
                    
                    print('learning rate decays to',newlr)

            #输出
            print('epoch:[{}],current_loss:[{}]'.format(epoch+1,avg_loss))
            print('耗时',(time.time()-start)/60,'min\n')
            
           
        torch.save(bestmodel,self.parameter_path)
        print('训练结束，总耗时：',(time.time()-start_time)/60)
        
    #用最好模型计算在验证集上的F2（cv）,batch_size为默认batch-size而不是128否则容易炸显存
    def eval(self,pre_trained=True):
        if pre_trained:#如果pretrained，则load
            self.model.load_state_dict(torch.load(self.parameter_path))#加载模型参数到预定路径中
        TP,FN,FP,TN=0,0,0,0
        loader_valid=DataLoader(dataset=self.dataset_valid,batch_size=self.batch_size,shuffle=False,num_workers=8)#关于droplast，若为true则舍掉最后个不完整batch
        remain_num = (len(self.dataset_valid) - 1) % self.batch_size + 1 #剩余部分,减一取模再加一，考虑整除情况 
        ones=torch.ones(self.batch_size,self.num_classes).to(self.device)
        zeros=torch.zeros(self.batch_size,self.num_classes).to(self.device)
        ones_remain=torch.ones(remain_num,self.num_classes).to(self.device)#最后一部分剩余量的掩码比较
        zeros_remain=torch.zeros(remain_num,self.num_classes).to(self.device)
        threshold_list=[]#用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i+1] - self.bound[i])
        threshold_tensor=torch.Tensor([threshold_list] * self.batch_size).to(self.device)

        with torch.no_grad():#避免梯度积累
            for iteration,(vectors,labels) in enumerate(loader_valid):
                if iteration==len(loader_valid)-1:
                    ones,zeros=ones_remain,zeros_remain
                    threshold_tensor = torch.Tensor([threshold_list] * remain_num).to(self.device) 
                vectors=vectors.to(self.device)#放在gpu上,此时的size是batch*5*channels*height*width
                labels=labels.to(self.device)#放在gpu上
                outputs=self.model.forward(vectors)#shape:batch_size*num_classes，此时只是得到线性映射结果
                outputs=torch.sigmoid(outputs)#sigmoid非线性激活
                
                outputs=outputs>threshold_tensor#二分为0-1矩阵
                TP+=((outputs==ones)&(labels==ones)).sum().item()
                FN+=((outputs==zeros)&(labels==ones)).sum().item()
                FP+=((outputs==ones)&(labels==zeros)).sum().item()
                TN+=((outputs==zeros)&(labels==zeros)).sum().item()
        P,R=TP/(TP+FP),TP/(TP+FN)
        F2=5*P*R/(P*4+R)
        print(self.model_name,'F2',F2,'Precison:',P,'Recall:',R)
        return F2
    
    #在测试集上的输出
    def inference(self):#用64作为测试时候batchsize
        loader_test = DataLoader(dataset=self.dataset_test, batch_size=64, shuffle=False,num_workers=8)
        test_id_data= pd.read_csv("/home/wzr/imet/sample_submission.csv")
        threshold_list=[]#用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i+1] - self.bound[i])#len ：3474
        threshold_tensor=torch.Tensor([threshold_list] * len(self.dataset_test)).to(self.device)#len(test)*num_classes

        preds = None
        with  torch.no_grad():
            for (iteration,images) in enumerate(loader_test):#在每个batch上连续经过stack1和stack2
                inputs_for_stack_2 = None#每个batch最终得到一个stack2的输入，再从stack2里得到一个输出
                images = images.to(self.device)
                for model_name in ['resnet101','se_resnext50','se_resnext101']: #三个模型 
                    model =self.model_choice(model_name).to(self.device)
                    avg_outputs = torch.zeros((len(self.dataset_test),3474)).to(self.device) #记录单模型的平均结果
                    for fold in range(6):
                        parameter_path = self.data_dir + model_name + str(fold) + '.pth'
                        model.load_state_dict(torch.load(parameter_path))
                        outputs = model.forward(images)
                        outputs = torch.sigmoid(outputs)
                        avg_outputs += outputs#六折求和

                    #此时current_preds应当为单模型的预测结果
                    avg_outputs /= 6#取六折的平均结果,batch*3474
                    if inputs_for_stack_2 == None:
                        inputs_for_stack_2 = avg_outputs#batch*3474
                    else:inputs_for_stack_2 = torch.cat((inputs_for_stack_2,avg_outputs),dim = 1)#64*10442
                inputs_for_stack_2 = torch.reshape(64,3,18,193)
                outputs_for_stack_2 = self.model.forward(outputs_for_stack_2)#64（或者remain）*3474
                outputs_for_stack_2 = torch.sigmoid(outputs_for_stack_2)
                if iteration == 0:
                    preds = outputs_for_stack_2
                else:preds = torch.cat((preds,outputs_for_stack_2),dim = 0)#len(dataset_test)* 3474
        
        predictions = preds.to(self.device) > threshold_tensor # 拼接成0/1矩阵
        # 替换原sample_submission.csv中的attribute_ids列
        for (i, row) in enumerate(predictions.to('cpu').numpy()):
            ids = np.nonzero(row)[0] # 把0/1矩阵的每一行变为非零元素的索引值数组
            test_id_data.iloc[i].attribute_ids = ' '.join([str(x) for x in ids]) # 空格连接，替换列
        test_id_data.to_csv('submission.csv', index=False) # 导出csv
        test_id_data.head() # 输出csv的前4行



            