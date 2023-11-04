# 照着改呗，还能怎样
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
from utils_RNN_BERT import Get_data   # 改成我的
from torch.autograd import Variable
from models import BGRU_BERT_cat    # 具体到那个函数
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('mel.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="BGRU_BERT_cat_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=30, metavar='N')   #改batch的意义就是改进多少次，batch越小，1个epoch里就反向传播次数越多
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=30)   #默认30
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=520)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)     # 执行四分类任务
parser.add_argument('--cnn_hangshu', type=int, default=128)  # 这里定义输入图片的行数是80px
args = parser.parse_args()
'''
训练参数：
--cuda: 使用GPU
--batch_size：training batch 
--dropout：
--epochs： training times
GRU参数：
--bid_flag: 
--batch_first:
--dia_layers
--out_class
Padding:
--utt_insize : 务必与谱信息的dim对应。
'''
torch.manual_seed(args.seed)   # 使得全局的随机数都是这个随机数种子，从而保证结果的可复现性

data_save = "BGRU_BERT_cat.txt"

# 定义岛损
class IslandLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(IslandLoss, self).__init__()
        self.alpha = alpha  # 类内损失的权重
        self.beta = beta  # 类间损失的权重

    def forward(self, embeddings, labels):
        # 计算类内损失
        intra_loss = 0
        for label in labels.unique():
            mask = (labels == label).nonzero().squeeze()
            if mask.numel() > 1:
                class_embeddings = embeddings[mask]
                mean_embedding = class_embeddings.mean(dim=0)
                intra_loss += F.mse_loss(class_embeddings, mean_embedding.expand_as(class_embeddings))

        # 计算类间损失
        inter_loss = 0
        num_classes = labels.unique().numel()
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                mask_i = (labels == i).nonzero().squeeze()
                mask_j = (labels == j).nonzero().squeeze()
                class_i_embeddings = embeddings[mask_i]
                class_j_embeddings = embeddings[mask_j]
                inter_loss += F.mse_loss(class_i_embeddings.mean(dim=0), class_j_embeddings.mean(dim=0))

        # 组合类内和类间损失
        total_loss = self.alpha * intra_loss + self.beta * inter_loss

        return total_loss

# 定义焦损
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def Train(epoch):
    train_loss = 0
    cnet.train()
    total_samples = 0
    correct_samples = 0
    for batch_idx, (data_1, target, id) in enumerate(train_loader):
        id = list(id)
        if args.cuda:
            data_1, target  = data_1.cuda(), target.cuda()

        utt_optim.zero_grad()
        target = target.squeeze()
        cout = cnet(data_1, id, args)
        loss = torch.nn.CrossEntropyLoss()(cout, target.long())
        # loss = IslandLoss()(cout, target.long())  #看看岛损的情况，很差，识别率25
        # loss = FocalLoss()(cout, target.long())  #看看焦损的情况

        loss.backward()

        utt_optim.step()
        train_loss += loss
        _, predicted = torch.max(cout, 1)
        total_samples += target.size(0)
        correct_samples += (predicted == target).sum().item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.2f}%'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval,
                       (correct_samples / total_samples) * 100
            ))
            train_loss = 0


def Test():
    cnet.eval()

    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx, (data_1, target ,id) in enumerate(test_loader):
            if args.cuda:
                data_1, target = data_1.cuda(), target.cuda()
            data_1, target = Variable(data_1), Variable(target)
            utt_optim.zero_grad()
            # data_1 = torch.squeeze()
            model_out = cnet(data_1 ,id, args)
            output = torch.argmax(model_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        # print(label_true)
        # print(label_pre)

        # 这里的函数是来自sklearn的，用于计算召回值，f1值，混淆矩阵
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true, CM_test


Final_result = []
Fineal_f1 = []
kf = KFold(n_splits=10)
sum_cm = np.zeros((4, 4))
for index, (train, test) in enumerate(kf.split(data)):
    print(f"开启第{index+1}折的训练 ：")  # 所以是有10大轮的训练构成。。。。
    train_loader, test_loader, input_test_data_id, input_test_label_org, input_train_data_id = Get_data(data, train, test, args)  # 加上了训练元素的id
    cnet = BGRU_BERT_cat(args.cnn_hangshu, args.hidden_layer, args)
    if args.cuda:
        cnet = cnet.cuda()

    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(cnet.parameters(), lr=lr)
    utt_optim = optim.Adam(cnet.parameters(), lr=lr)
    f1 = 0
    recall = 0
    recall_list = []
    f1_list = []
    cm_list = []
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        # 加上最佳值的输出保存
        accuracy_f1, accuracy_recall, pre_label, true_label, cm = Test()
        # 把数据构建成列表
        recall_list.append(accuracy_recall)
        f1_list.append(accuracy_f1)
        cm_list.append(cm)

        if epoch % 15 == 0:
            lr /= 10
            for param_group in utt_optimizer.param_groups:
                param_group['lr'] = lr

        if (accuracy_f1 > f1 and accuracy_recall > recall):
            name_1 = 'BGRU_BERT_AttF' + str(index) + '.pkl'
            torch.save(cnet.state_dict(), name_1)
            recall = accuracy_recall
            f1 = accuracy_f1
    # 跑完一折就保存最好数据
    # 通过recall来锁定最大值
    max_recall = max(recall_list)
    max_f1 = f1_list[recall_list.index(max_recall)]  # 通过在recall列表里检索下标来输出对应的f1数值
    cm = cm_list[recall_list.index(max_recall)]
    sum_cm += cm
    print("成功统计一个混淆矩阵")
    with open(data_save, 'a') as f:
        f.write("第" + str(index + 1) + "折数据：" + "\n" + str(max_recall) + '\n' + str(max_f1) + '\n' + str(cm) + '\n')
        print("输出结果已保存")
with open(data_save, 'a') as f:
    f.write('\n10个最佳混淆矩阵之和是：\n'+str(sum_cm))
    print("最终混淆矩阵结果已保存")