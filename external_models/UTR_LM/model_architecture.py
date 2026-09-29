"""
模型架构定义
直接从 MJ4_Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py 提取
保持与训练脚本完全一致
"""

import os
import sys
import torch.nn as nn

# 添加路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'Scripts'))

from esm.model.esm2_supervised import ESM2
from esm.data import Alphabet


class CNN_linear(nn.Module):
    """
    TE 预测模型
    从 MJ4_Finetune_extract_append_predictor_CellLine_10fold-lr-huber-DDP.py
    line 120-217 提取
    """

    def __init__(self,
                 layers=6,
                 heads=16,
                 embed_dim=128,
                 inp_len=100,
                 nodes=40,
                 dropout3=0.2,
                 cnn_layers=0,
                 avg_emb=False,
                 bos_emb=True,
                 magic=False,
                 modelfile='ESM2SI',
                 border_mode='same',
                 filter_len=8,
                 nbr_filters=120,
                 dropout1=0,
                 dropout2=0):

        super(CNN_linear, self).__init__()

        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = nodes
        self.cnn_layers = cnn_layers
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.layers = layers
        self.avg_emb = avg_emb
        self.bos_emb = bos_emb
        self.magic = magic

        # 创建 alphabet
        alphabet = Alphabet(standard_toks='AGCT', mask_prob=0.0)

        # ESM2 backbone
        if 'SISS' in modelfile or 'SI' in modelfile:
            # 使用监督学习版本
            self.esm2 = ESM2(
                num_layers=layers,
                embed_dim=embed_dim,
                attention_heads=heads,
                alphabet=alphabet
            )
        else:
            self.esm2 = ESM2(
                num_layers=layers,
                embed_dim=embed_dim,
                attention_heads=heads,
                alphabet=alphabet
            )

        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=self.embedding_size,
            out_channels=self.nbr_filters,
            kernel_size=self.filter_len,
            padding=self.border_mode
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.nbr_filters,
            out_channels=self.nbr_filters,
            kernel_size=self.filter_len,
            padding=self.border_mode
        )

        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully connected layers
        if avg_emb or bos_emb:
            self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        else:
            self.fc = nn.Linear(in_features=inp_len * embed_dim, out_features=self.nodes)

        if avg_emb or bos_emb:
            self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        else:
            self.linear = nn.Linear(in_features=inp_len * self.nbr_filters, out_features=self.nodes)

        self.output = nn.Linear(in_features=self.nodes, out_features=1)

        if self.cnn_layers == -1:
            self.direct_output = nn.Linear(in_features=embed_dim, out_features=1)
        if magic:
            self.magic_output = nn.Linear(in_features=1, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation=True):
        """
        前向传播
        完全复制自 MJ4 训练脚本 line 177-217
        """
        x = self.esm2(tokens, [self.layers], need_head_weights, return_contacts, return_representation)

        if self.avg_emb:
            x = x["representations"][self.layers][:, 1 : self.inp_len+1].mean(1)
            x_o = x.unsqueeze(2)
        elif self.bos_emb:
            x = x["representations"][self.layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x["representations"][self.layers][:, 1 : self.inp_len+1]
            x_o = x_o.permute(0, 2, 1)

        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2:
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3:
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)

        x = self.flatten(x_o)

        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)

        if self.magic:
            o = self.magic_output(o)

        return o
