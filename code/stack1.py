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

from dataset import MyDataset_stack1,trans_train,trans_valid
#import models
import models


#定义分类器与其各功能
class Stack1(nn.Module):
    def __init__(self,num_classes,lr,device,num_epoches,batch_size,threshold,model_name,data_dir,valid_fold,weight_dacay):
        super(Stack1,self).__init__()
        
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

        
        #模型选择
        self.model =self.model_choice(model_name)
        
        self.criterion=nn.BCEWithLogitsLoss(reduction='mean')#对最后一层(线形层之后的结果)每个神经元做逻辑斯蒂回归
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=weight_dacay)#第一个阶梯的lr

        #数据导入
        self.dataset_train,self.dataset_valid,self.dataset_test=self.get_dataset()
        self.parameter_path=data_dir+model_name+ str(self.valid_fold) +'.pth'#生成参数的保存路径

    #用于导入数据并返回对应的dataset对象,传入数据只有路径
    #分折训练32*740*6=142080（总数据量为142119）
    def get_id_data(self):

        labels= pd.read_csv(self.data_dir+"labels.csv")
        train_id_data= pd.read_csv(self.data_dir+"train.csv")
        test_id_data= pd.read_csv(self.data_dir+"sample_submission.csv")

        y =train_id_data.attribute_ids.map(lambda x: x.split()).values   #将attribute每个值进行划分，因为一个值包含多个标签
        train_id_data['y']= y  #:增加纵向列为label(int)的lsit
        test_id_data= test_id_data['id']

        unfold_train_id_data = train_id_data.values#用values可以取到ndarray of (id,attribute_ids(为string类型),list of (attribute_ids))
        test_id_data = test_id_data.values#用values可以取到ndarray of id
        
        return unfold_train_id_data,test_id_data
    #得到dataset类
    def get_dataset(self):
         
        train_path= self.data_dir+'train/'
        test_path= self.data_dir+'test/'
        unfold_train_id_data,test_id_data =self.get_id_data()#提取id_data为ndarray
        #将多余的数据补充到最后一折中
        if self.valid_fold == 5:
            valid_id_data = unfold_train_id_data[23680 * 5 : ]
            train_id_data = unfold_train_id_data[0 : 23680 * 5]
        else: 
            valid_id_data = unfold_train_id_data[23680 * self.valid_fold : 23680 * (self.valid_fold + 1)]
            train_id_data = np.vstack((unfold_train_id_data[0:23680 * self.valid_fold], unfold_train_id_data[23680*(self.valid_fold+1):]))
        #train_id_data,valid_id_data = train_test_split(train_id_data,test_size=0.1)
        #传入的df_data已经是values
        dataset_train= MyDataset_stack1(df_data=train_id_data,mode='train',data_dir=train_path,transform=trans_train)
        dataset_valid= MyDataset_stack1(df_data=valid_id_data, mode='train', data_dir=train_path,transform=trans_valid)
        dataset_test= MyDataset_stack1(df_data=test_id_data, mode='test', data_dir=test_path,transform=trans_valid)
        print('导入数据结束')
        return dataset_train,dataset_valid,dataset_test
    
    #参数训练
    def train(self,pre_trained = False, min_loss = np.inf):
        loader_train=DataLoader(dataset=self.dataset_train,batch_size=self.batch_size,shuffle=True,num_workers=8)
        loader_valid=DataLoader(dataset=self.dataset_valid,batch_size=self.batch_size,shuffle=False,num_workers=8)
        min_val_loss = min_loss#记录最小loss
        sym,times=0,0#用来控制学习率的衰减,记录不减loss数和衰减次数
        if pre_trained:self.model.load_state_dict(torch.load(self.parameter_path))
        bestmodel=self.model.state_dict()#用来记录模型参数
        start_time=time.time()
        print('start training valid_fold',self.valid_fold)

        for epoch in range(self.num_epoches):
            start=time.time()
            avg_loss = 0
            #训练
            for iteration,(images,labels) in enumerate(loader_train):
                images=images.to(self.device)#shape:batch*in_channels*height*width
                labels=labels.to(self.device)#shape:batch*num_classes
                self.optimizer.zero_grad()
                outputs=self.model.forward(images)
                loss=self.criterion(outputs,labels)
                loss.backward()
                self.optimizer.step()
                #if iteration%200==0:print('epoch', epoch,' iteration',iteration,'finished ,taking',(time.time()-start)/60,'min')
            
            #在验证集上计算loss和f2
            current_F2 = self.eval(pre_trained=False)#计算f2，已在内部断开梯度累计
            with torch.no_grad():
                for i,(images, labels) in enumerate(loader_valid):
                    images = images.to(self.device)#shape:batch*in_channels*height*width
                    labels = labels.to(self.device)#shape:batch*num_classes
                    outputs = self.model.forward(images)#shape:batch*num_calsses
                    loss = self.criterion(outputs, labels)
                    avg_loss += loss.item() /len(loader_valid)
            
            #通过patience实现衰减机制
            if min_val_loss > avg_loss:
                bestmodel = self.model.state_dict()#以loss为标准保存最好模型
                torch.save(bestmodel,self.parameter_path)#每次都保存模型
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
    def eval(self,pre_trained=False):
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
            for iteration,(images,labels) in enumerate(loader_valid):
                if iteration==len(loader_valid)-1:
                    ones,zeros=ones_remain,zeros_remain
                    threshold_tensor = torch.Tensor([threshold_list] * remain_num).to(self.device) 
                images=images.to(self.device)#放在gpu上,此时的size是batch*5*channels*height*width
                labels=labels.to(self.device)#放在gpu上
                outputs=self.model.forward(images)#shape:batch_size*num_classes，此时只是得到线性映射结果
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

    # 实现在测试集上的输出和csv文件生成
    def inference(self,models_name,fold_list):#avg表示用多折混合模型平均预测
        # 载入测试集数据
        loader_test = DataLoader(dataset=self.dataset_test, batch_size=128, shuffle=False,num_workers=8)
        test_id_data= pd.read_csv(self.data_dir+"sample_submission.csv")
        threshold_list=[]#用于构造界的tensor
        for i in range(5):
            threshold_list += [self.threshold[i]] * (self.bound[i+1] - self.bound[i])#len ：3474
        threshold_tensor=torch.Tensor([threshold_list] * len(self.dataset_test)).to(self.device)#len(test)*num_classes

        # 开始测试
        preds = torch.zeros((len(self.dataset_test),3474)).to(self.device)#保存计算结果
        for model_name in models_name:
            model = self.model_choice(model_name)#选择模型
            if model_name == 'se_resnet50':
                data_dir = '../input/se-resnet50/'
            if model_name == 'se_resnext50':
                data_dir = '../input/se-resnext50/'
            for fold in fold_list:
                single_preds = None#记录单模型的单折结果
                parameter_path = data_dir+model_name+str(fold)+'.pth'#加载对应折训练参数
                model.load_state_dict(torch.load(parameter_path))
                for (i, images) in enumerate(loader_test):
                    images = images.to(self.device)
                    with torch.no_grad():
                        y_preds = model.forward(images)
                        if i == 0:
                            single_preds = torch.sigmoid(y_preds)#第一次
                        else:
                            single_preds= torch.cat((single_preds,torch.sigmoid(y_preds)),dim = 0)#在0维度上拼接
                preds += single_preds#做累加
            preds /= len(fold_list)#每折的平均结果
        preds /= len(models_name)#每个模型的平均结果

        # 测试结束

        # 生成submission.csv
        predictions = preds.to(self.device) > threshold_tensor # 拼接成0/1矩阵
        # 替换原sample_submission.csv中的attribute_ids列
        for (i, row) in enumerate(predictions.to('cpu').numpy()):
            ids = np.nonzero(row)[0] # 把0/1矩阵的每一行变为非零元素的索引值数组
            test_id_data.iloc[i].attribute_ids = ' '.join([str(x) for x in ids]) # 空格连接，替换列
        test_id_data.to_csv('submission.csv', index=False) # 导出csv
        test_id_data.head() # 输出csv的前4行
    
    def model_choice(self,model_name):
        if model_name=='se_resnet50':
            return models.se_resnet50(num_classes=3474).to(self.device)
        if model_name=='se_resnext50':
            return models.se_resnext50(num_classes=3474).to(self.device)
        if model_name=='resnext50':
            return models.resnext50(num_classes=3474).to(self.device)
        if model_name=='airnext50':
            return models.airnext50(num_classes=3474).to(self.device)
        if model_name=='se_resnet101':
            return models.se_resnet101(num_classes=3474).to(self.device)
        if model_name=='se_resnext101':
            return models.se_resnext101(num_classes=3474).to(self.device)
        if model_name =='resnext101':
            return models.resnext101(num_classes=3474).to(self.device)

    #实现在train上对stacking特征的输出,用多个model_name
    def predict(self):
        train_id_data,test_id_data =self.get_id_data()#得到id数据，ndarray，用于分折预测train的结果
        preds = None
        train_path= self.data_dir+'train/'#得到数据路径
        for model_name in ['resnet101','se_resnext50','se_resnext101']: #三个模型
            current_preds=None#记录单模型的拼接结果
            model = self.model_choice(model_name).to(self.device)#选择模型
            for fold in range(6):#根据fold选择测试集以及模型参数
                if fold == 5: valid_id_data = train_id_data[23680 * 5 : ]
                else:valid_id_data = train_id_data[23680 * fold : 23680 * (fold + 1)]#得到valid集合
                dataset_valid= MyDataset_stack1(df_data=valid_id_data, mode='train', data_dir=train_path,transform=trans_valid)
                loader_valid=DataLoader(dataset=dataset_valid,batch_size=128,shuffle=False,num_workers=8)#得到batch化的数据，用128做为batch合并次数减少
                parameter_path =self.data_dir+model_name+str(fold)+'.pth'#根据模型和fold选择模型参数
                model.load_state_dict(torch.load(parameter_path))
                    
                #进行分折预测并拼接结果
                with torch.no_grad():#避免梯度积累
                    for iteration,(images,labels) in enumerate(loader_valid):
                        images=images.to(self.device)#放在gpu上,此时的size是batch*5*channels*height*width
                        labels=labels.to(self.device)#放在gpu上
                        outputs=model.forward(images)#shape:batch*3474,用对应的model得到结果
                        outputs=torch.sigmoid(outputs)#sigmoid非线性激活
                        outputs *=0.7#继承0.7的teacher数据
                        outputs =torch.max(outputs,labels)#再与labels取最大
                        if iteration==0 and fold ==0:current_preds =outputs#第一折，第一次
                        else:current_preds = torch.cat((current_preds,outputs),dim = 0)#在第0维度上拼接

            #此时current_preds应当为单模型的预测结果
            print(current_preds.size())#打印规模结果,应当是14w*3474
            if preds == None: preds = current_preds#14w*3474
            else: preds = torch.cat((preds,current_preds),dim = 1)#在第1维度上拼接：14w*10422
        preds = torch.transpose(preds,dim0=0, dim1 = 1)#转置，以方便pandas读取
            
        print(preds.size()) 
        preds =preds.to('cpu').numpy()
        np.savetxt(self.data_dir + 'train_stack.csv',preds,delimiter=',')
               
                
            


                    


            
            

                      
            
  
            








            










        


