import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from torch.autograd import Variable
# from models import Utterance_net
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Label,id):
        self.Data_1 = Data_1
        self.Label = Label
        self.Id = id
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = torch.Tensor(self.Data_1[item])
        label = torch.Tensor(self.Label[item])
        id = self.Id[item]
        return data_1,label ,id

def Feature(data,args):
    input_data_spec = []
    for i in range(len(data)):
        input_data_spec.append(data[i]['gram_data'])
    a = [0.0 for i in range(args.cnn_hangshu)]
    input_data_spec_CNN = []
    for i in range(len(input_data_spec)):
        ha = []
        if(len(input_data_spec[i]) < 300):
            for z in range(len(input_data_spec[i])):
                ha.append(input_data_spec[i][z])
            len_zero = 300 - len(input_data_spec[i])
            for x in range(len_zero):
                ha.append(a)       # 补上几列的80行
        if(len(input_data_spec[i]) >= 300):
            for z in range(len(input_data_spec[i])):
                if(z < 300):
                    ha.append(input_data_spec[i][z])
        ha = np.array(ha)
        # ha = ha.reshape((1, 300, 128))  # 把(300,80)变成(1,300,80)   mel的话80换成128
        input_data_spec_CNN.append(ha)

    # 看看这里的特征是什么形状
    # input_data_spec_CNN = np.array(input_data_spec_CNN)
    # print("input_spec_CNN:",input_data_spec_CNN.shape)        input_spec_CNN: (512, 300, 80)

    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label_cat'])

    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'])
    # print(input_data_id)    诸如此类的：'Ses05F_script01_3_M000', 'Ses05F_impro06_M011'

    input_label_org = []
    for i in range(len(data)):
        input_label_org.append(data[i]['label_cat'])
    return input_data_spec,input_data_spec_CNN,input_label,input_data_id,input_label_org

def Get_data(data,train,test,args):
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])
    for i in range(len(test)):
        test_data.extend(data[test[i]])

    # 前三个分别是：完整的频谱图，300定长频谱图（截断、补零），标签
    input_train_data_spec,input_train_data_spec_CNN,input_train_label,input_train_data_id,_ = Feature(train_data,args)
    # print(np.shape(input_train_data_spec_CNN))  # (4961, 1, 300, 80)
    # print(np.shape(input_data_id))              # (4961,)

    input_test_data_spec,input_test_data_spec_CNN, input_test_label,input_test_data_id,input_test_label_org = Feature(test_data,args)


    # label = np.array(input_train_label, dype='int64').reshape(-1,1)
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)

    # 分别加入了两个data_id
    train_dataset = subDataset(input_train_data_spec_CNN,label, input_train_data_id)
    test_dataset = subDataset(input_test_data_spec_CNN,label_test, input_test_data_id)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=True, shuffle=False)
    return train_loader,test_loader,input_test_data_id,input_test_label_org, input_train_data_id