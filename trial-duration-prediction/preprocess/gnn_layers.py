import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
torch.manual_seed(3) 
np.random.seed(1)

#一个图卷积层 (Graph Convolutional Layer, GCN)，其功能是基于图结构数据进行卷积操作
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        # in_features: 输入特征的维度。
        # out_features: 输出特征的维度。
        # bias: 是否使用偏置项，默认值为 True。
        # init: 权重初始化方法，默认为 xavier
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 根据传入的初始化方法 ('uniform', 'xavier', 'kaiming')，选择不同的初始化策略来初始化权重。
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    # input: 输入特征矩阵，形状为 (num_nodes, in_features)，即每个节点的特征。
    # adj: 图的邻接矩阵，形状为 (num_nodes, num_nodes)，表示图中节点的连接关系。
    # support：将输入特征矩阵 input 和权重矩阵 self.weight 相乘，计算得到节点的支持特征。
    # output：通过邻接矩阵 adj 对 support 特征进行图卷积操作，得到每个节点的输出特征。
    def forward(self, input, adj):
        support = torch.mm(input, self.weight) #通过矩阵乘法 torch.mm(input, self.weight) 来变换输入特征矩阵（input）
        output = torch.spmm(adj, support) #通过稀疏矩阵乘法 torch.spmm(adj, support) 计算卷积操作，其中 adj 是邻接矩阵，support 是每个节点的特征变换结果。邻接矩阵用于聚合邻居节点的特征。
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#实现了一个图注意力层（GAT layer），引入了自注意力机制，允许模型为不同的邻居节点分配不同的权重。
class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        # in_features: 输入特征的维度（每个节点的输入特征的大小）。
        # out_features: 输出特征的维度（每个节点输出的特征的大小）。
        # dropout: dropout的概率，用于防止过拟合。
        # alpha: LeakyReLU 激活函数的负半轴斜率。
        # concat: 是否将多头注意力的输出拼接。
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        #self.W: 权重矩阵，用于线性变换输入特征。通过 Xavier 初始化进行初始化。
        # self.a1 和 self.a2: 用于计算注意力分数的权重，分别对输入特征的不同表示进行变换并生成注意力系数。这里使用了 Xavier 初始化。
        # nn.Parameter 使得这些权重矩阵成为可训练参数。
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # input: 输入特征矩阵，形状为 (num_nodes, in_features)，每行是一个节点的特征。
        # adj: 邻接矩阵，表示图中节点之间的连接关系。
        # h: 通过矩阵乘法将输入特征矩阵 input 和权重矩阵 self.W 相乘，得到变换后的节点特征。
        # f_1 和 f_2: 分别计算节点特征的线性变换，用于生成注意力系数。
        # e: 使用 LeakyReLU 激活函数计算注意力系数。f_2.transpose(0, 1) 是为了保证矩阵尺寸匹配（节点间的相似度计算）。
        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        # zero_vec: 用于对不相邻的节点赋予一个非常小的值，防止它们的注意力系数为正数。
        # attention: 根据邻接矩阵 adj 来选择计算注意力系数。只有相邻节点之间的注意力系数 e 会被保留，否则使用一个很小的值（zero_vec）。
        # F.softmax: 对每个节点的邻居节点进行归一化处理，使得注意力系数的和为1。
        # F.dropout: 在训练过程中对注意力系数进行 dropout 操作。
        # h_prime: 计算节点的新特征表示，通过加权的邻接节点特征。

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



