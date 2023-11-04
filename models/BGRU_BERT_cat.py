from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
import re

#隔壁的bert训练模型还没同步过来，如果效果好就同步过来
# 指定本地模型路径 这两个是服务bert用的
model_path = "bert-base-uncased"
# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(model_path)
MAX_LEN = 128
# 读取CSV文件
data_frame = pd.read_csv("IEMOCAP_sentence_trans.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BGRU_BERT_cat(nn.Module):
    def __init__(self, input_size, hidden_size, args):
        super(BGRU_BERT_cat, self).__init__()
        ##CNN部分##
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 100)  # 输出100个元素的一维矩阵，bert那边也出100个(原本是768，线性成100)，两者占比一半一半
        )
        ##CNN结束## 留着作对比（如果有必要）

        ##GRU部分##
        self.hidden_dim = args.hidden_layer
        self.num_layers = args.dia_layers
        self.dropout = nn.Dropout(args.dropout)
        self.bigru = nn.GRU(input_size, self.hidden_dim, dropout=args.dropout,
                           batch_first=True, num_layers=self.num_layers, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, 256)   #GRU出来[batch,256]
        ##GRU结束##

        self.bert = BertModel.from_pretrained(model_path)  # 出来[batch,768]
        self.classifier = nn.Sequential(
            nn.Linear(1024,512),
            nn.Linear(512,256),
            nn.Linear(256,4)
        )



    def forward(self, x, spec_id, args):
        ##BERT处理部分##
        input_ids_list = []
        attention_mask_list = []
        for item in spec_id:
            # 查询给定编号对应的文本和标签
            result = data_frame[data_frame['id'] == item]
            if len(result) == 0:
                return None, None
            text = result['transcription'].values[0]
            # 使用正则表达式去除标点符号
            text = re.sub(r'[^\w\s]', '', text)

            # 步骤3: 对输入文本进行tokenization并添加特殊token
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN,
                                 return_tensors='pt')

            input_ids = encoding.get('input_ids').squeeze().to(device)  # 把[1,128]变成[128]
            attention_mask = encoding.get('attention_mask').squeeze().to(device)  # 把[1,128]变成[128]

            # 把两位append进列表中，等待下一步的堆叠
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        input_ids = torch.cat(input_ids_list, dim=0).view(args.batch_size, MAX_LEN)  # 堆叠成[batch,MAX_LEN]
        attention_mask = torch.cat(attention_mask_list, dim=0).view(args.batch_size, MAX_LEN)  # 堆叠成[batch,MAX_LEN]

        output = self.bert(input_ids, attention_mask)  # 改换思路，把矩阵丢进bert
        BERT_out = output.pooler_output  # 拿取pooler层的输出 ,large是[batch,1024]和base是[batch,768]
        # BERT_out = self.FC1(pool_out)   # bert出来[batch,100]
        ##BERT处理结束##

        ##GRU处理部分##
        embed = self.dropout(x)
        # gru
        gru_out, _ = self.bigru(embed)
        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = torch.tanh(gru_out)
        # linear
        GRU_out = self.linear1(gru_out)   #GRU处理的 spec图的[128,100]
        ##GRU处理结束##

        ##开始拼接##
        GRU_BERT = torch.cat([BERT_out,GRU_out],dim = 1)  # [batch,768+256]

        x = self.classifier(GRU_BERT)

        return x