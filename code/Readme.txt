country&dimension.py   ：运行得到country&dimension_Stat.csv
drawcloud.py：运行得到词云图
label&num.py：运行得到label-num.csv
labelPerPic.py：运行得到country&dimension_Stat.csv

dataset.py
主要功能：实现数据加载和预处理函数
具体函数和类介绍：
trans_train:训练集上的图片（img类）->tensor转换与数据增强
trans_valid:验证集和测试集上的图片（img类）->tensor转换与数据增强
MyDataset类:实现了通过路径加载图片并整合成可迭代对象的类，与torch内置包dataloader兼容

models.py
主要功能：实现各类model，包括各类se和attention机制的resnet和airnext
具体函数功能详见注释介绍

stack1.py
主要功能：定义第一层分类器与其各功能
具体成员函数的介绍：
Get_dataset:获取标签数据和路径数据并构建为dataset类
Train:训练模型并在验证集上测试
Eval:计算验证集的F2:
inference:模型融合的逻辑和写法、在测试集（test）上的输出、kaggle交互提交

main.py
设置超参数

stack2.py 实现第二层stack模型，功能与stack1包含的类似

dataset_fivecrop.py用fivecrop作为数据预处理部分代码 

analysis.py
在原模型的基础上添加了analysis, getpic, ext_spe_att三个模块，并且在原有代码中做出了一些小的修改。
analysis主要作用是使用给点模型对训练集前五百张图片进行预测，将预测结果打印成文本文档以供分析
getpic模块主要作用在导出给定id的图片，是一个轻量级的接口
ext_spe_att主要作用在对给定模型的给定层数据导出，以实现中间过程的数据可视化。由于正则化工作并不理想，此模块被弃用