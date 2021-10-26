# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:24:27 2020
@author: Justin
"""
'''
统计每张图片country和dimension的数目
'''
from pyecharts.charts import WordCloud


src_filename = './train.csv'

#flag=True
src_file = open(src_filename, 'r')
line_list = src_file.readlines()  #返回列表，文件中的一行是一个元素
src_file.close()
del line_list[0]

output_filename = './country&dimension_Stat.csv'
output_file=open(output_filename,'w')

for line in line_list:
    line = line.strip()  #删除'\n'
    for i in range(len(line)):
        if line[i] == ',':
            picID = line[0:i]
            labelset = line[i+1:]
            break;
    line_split = labelset.split(' ')
    #if flag:
     #   print(line_split)
      #  flag=False
    countryNum=0
    dimensionNum=0
    cultureNum = 0
    mediumNum = 0
    tagsNum=0
    for i in range(len(line_split)):
        labelNum=int(line_split[i])
        if labelNum in range(0,100):
            countryNum=countryNum+1
        elif labelNum in range(100,781):
            cultureNum = cultureNum +1
        elif labelNum in range(781,786):
            dimensionNum = dimensionNum +1
        elif labelNum in range(786,2706):
            mediumNum = mediumNum +1
        elif labelNum in range(2706,3474):
            tagsNum = tagsNum + 1
    writeback = picID +','+ str(countryNum)+','+str(cultureNum)+','+str(dimensionNum)+','+str(mediumNum)+','+str(tagsNum)+'\n'
    output_file.write(writeback)
    
output_file.close()

#output=open('./stat.csv','w')
#for i in list(labelStatis.keys()):
#    output.write(str(i)+','+str(labelStatis[i])+'\n')
#output.close()

#output=open('./labelStat.csv','w')
#for i in list(labelnum.keys()):
#    output.write(str(i)+','+str(labelnum[i])+'\n')
#output.close()

#print(wordfreq_list)
#del wordfreq_list[0] #删除csv文件中的标题行
##-------从文件中读出人物词频完成------------------

##===============================================
##-------生成词云---------------------------------
#cloud = WordCloud() # 初始化词云对象

# 向词云中添加内容，第一个参数可以设为空，第二个参数为元组列表（词和词频）
#cloud.add('', wordfreq_list)

# render会生成HTML文件。默认是当前目录render.html，也可以指定文件名参数
#out_filename = './output/wordcloud_rd_file.html'
#cloud.render(out_filename)

#print('生成结果文件：' + out_filename)

