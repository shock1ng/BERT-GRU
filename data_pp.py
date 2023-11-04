import re
import pickle
import os
import operator
import numpy as np
import csv
from sklearn import preprocessing


name = 'mel'   # 把所有的，以id命名的csv放进这个文件夹中,比如mel、raw等
data_file = name
data_file_1 = 'IEMOCAP_10039.csv'   # 查表csv文件

def as_num(x):
    y = '{:.10f}'.format(x)
    return float(y)

def STD(input_fea,name):
    #标准化
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i][name])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    print(len(a[0]))
    print(len(a))
    scaler_1 = preprocessing.StandardScaler().fit(a)
    print(scaler_1.mean_)
    print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i][name] = scaler_1.transform(input_fea[i][name])
    return input_fea

gram_data_dir = data_file  # 此处是"mel" ,把所有的，以id命名的csv放进这个文件夹中
traindata = []

num = 0

for sess in os.listdir(gram_data_dir):  #sess 是单个csv文件的名称 ;  os.listdir只返回名称列表，不返回路径
    #提取谱信息
    data_dir = gram_data_dir + '/' + sess   # mel/xxxxx.csv
    data_1 = []
    data = {}
    file = open(data_dir,'r')
    file_content = csv.reader(file)    #按逗号为分隔读取csv
    for row in file_content:           # 逐行读取CSV文件内容，每行都是:数据,数据,数据,数据,...,数据，对于mel的csv来说，一共有128行
        x = []
        for i in range(len(row)):      # 读取行的长度
            x.append(float(row[i]))    # 把这一行的数据全部以浮点型存进列表x中
        row = np.array(x)
        data_1.append(row)             # 把这一行的数据存进data_1中，48-53行的循环结束后，data_1就有一个谱图的全部数据了
    data['id'] = sess[:-4]             # data是字典，把sess从列表的开头开始切片，一直到倒数第五个元素（不包括倒数第四个元素），也就是不包含后缀的四位(.csv)，如   Ses01F_impro01_F000 切 .csv
    data_1_1 = np.array(data_1)
    data['gram_data'] = data_1_1.T     # 把谱图矩阵转置后存入'gram_data' 中，如：目前字典中的数据是{'id':"Ses01F_impro01_F000",'gram_data':[谱图].......}
    num = num + 1                      # num计数+1
    traindata.append(data)             # 把字典存入traindata列表中，最终traindata有10039个字典在里面
print("已经提取了谱信息")

id_label = []
file = open(data_file_1,'r')   # 打开查表csv文件
file_content = csv.reader(file)
#提取标签
for row in file_content:
    if(row[0][0] == 'S'):                 # 如果首字母是S，则进行下列操作
        data = {}                         # 构建data字典
        data['id'] = row[0]
        data['label_cat'] = int(row[5])
        data['label_V'] = float(row[2])
        data['label_A'] = float(row[3])
        data['label_D'] = float(row[4])
        data['speaker'] = row[6]
        data['i/s'] = row[7][0]           # 是i还是s，取7列的首字母，7列的内容是(imp 和 scr)
        id_label.append(data)             # 把每个字典都存入列表中
print(len(id_label))
print("已经提取了10039.csv的信息")

label = [1,2,3,4,5]
#对齐标签-说话人-谱信息
input_fea = []
for i in range(len(traindata)):
    for j in range(len(id_label)):
        if(traindata[i]['id'] == id_label[j]['id']):     # 当两个文件的id对齐之后开始下面操作
            if(id_label[j]['label_cat'] in label):       # 只操作[1,2,3,4,5]的，别的情感标签都不考虑
                if(id_label[j]['i/s'] == 'i' or 's'):    # 在[1,2,3,4,5]内还要考虑i/s值，别的都不要
                    if(id_label[j]['label_cat'] == 5):
                        id_label[j]['label_cat'] = 2     #如果标签是5，就变成2
                    data = {}         # 创建一个新的字典
                    data['label_cat'] = id_label[j]['label_cat'] - 1   # 标签前移，因为分类任务的神经网络的标签必须是0开始的、按顺序的
                    data['label_V'] = id_label[j]['label_V']
                    data['label_A'] = id_label[j]['label_A']
                    data['label_D'] = id_label[j]['label_D']
                    data['speaker'] = id_label[j]['speaker']
                    data['id'] = traindata[i]['id']
                    data['gram_data'] = traindata[i]['gram_data']
                    input_fea.append(data)    # 现在全部数据都在input_fea里面了

input_fea = STD(input_fea,'gram_data')    # 把字典里面的全部谱图数据进行标准化
print("已经提对齐提取了信息")

speaker = ['1','2','3','4','5','6','7','8','9','10']
#按照说话人分折

num = 0
data = [[],[],[],[],[],[],[],[],[],[]]
for i in range(len(input_fea)):
    for j in range(len(speaker)):
        if(input_fea[i]['speaker'] == speaker[j]):   # 说话人对齐
            data[j].append(input_fea[i])   # 最终的data = [[说1数据],[说2数据],[说3数据],...]
            num = num +1
print(num)
print("已经提对齐说话人提取了信息")
file_name = name + '.pickle'
file = open(file_name, 'wb')
pickle.dump(data,file)
file.close()