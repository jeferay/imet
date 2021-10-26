# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:24:27 2020
@author: Justin
"""
'''
从CSV文件中读出词频
用pyecharts画出词云
'''
from pyecharts.charts import WordCloud

src_filename = './label-num.csv'
# 格式：标签,出现次数
flag=True
src_file = open(src_filename, 'r')
line_list = src_file.readlines()  #返回列表，文件中的一行是一个元素
src_file.close()
del line_list[0]

labelStatis = {}  #用于保存元组(标签,出现次数)
labelnum={}
for line in line_list:
    line = line.strip()  #删除'\n'
    for i in range(len(line)):
        if line[i] == ',':
            labelset = line[i+1:]
            break;
    line_split = labelset.split(' ')
    if flag:
        print(line_split)
        flag=False
    for i in range(len(line_split)):
        if line_split[i] in labelStatis:
            labelStatis[line_split[i]]=labelStatis[line_split[i]]+1
        else:
            labelStatis[line_split[i]]=1
    labelnums=len(line_split)
    if labelnums in labelnum:
        labelnum[labelnums]=labelnum[labelnums]+1
    else:
        labelnum[labelnums]=1
      


