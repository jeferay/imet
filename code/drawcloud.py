# -*- coding: utf-8 -*-

from pyecharts.charts import WordCloud

##-------从文件中读出标签信息------------------
src_filename = './label-num.csv'
flag=True
src_file = open(src_filename, 'r')
line_list = src_file.readlines()  #返回列表，文件中的一行是一个元素
src_file.close()


tuplelist = []  #用于保存元组

for line in line_list:
    line = line.strip()  #删除'\n'
    line_split = line.split(',')
    for i in range(len(line_split[0])):
        if line_split[0][i] == ':':
            labelname = line_split[0][i+2:]
            break;
    tuplelist.append((labelname,line_split[1]))
      


cloud = WordCloud() # 初始化词云对象

# 向词云中添加内容，第一个参数可以设为空，第二个参数为元组列表（词和词频）
cloud.add('', tuplelist,shape='triangle-forward')

# render会生成HTML文件。默认是当前目录render.html，也可以指定文件名参数
out_filename = './cloud.html'
cloud.render(out_filename)

#print('生成结果文件：' + out_filename)

