import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np 
sigmoid = torch.nn.Sigmoid() 
torch.manual_seed(0)

from preprocess.gnn_layers import GraphConvolution, GraphAttention
torch.manual_seed(4) 
np.random.seed(1)

#利用门控机制，允许模型在每一层的学习过程中，选择性地使用非线性变换（即f(G(x))）或保持原始输入（即Q(x)）。门控机制有助于解决深度神经网络中的梯度消失问题，因为它使得信息可以通过网络的层级进行更容易的传递。
class Highway(nn.Module):
    def __init__(self, size, num_layers):
        super(Highway, self).__init__()
        # size：输入特征的维度。
        # num_layers：要堆叠的层数。
        # self.nonlinear：一个包含num_layers个Linear层的列表，每一层用于非线性变换。
        # self.linear：一个包含num_layers个Linear层的列表，用于线性变换。
        # self.gate：一个包含num_layers个Linear层的列表，用于计算门控权重。
        # self.f：非线性激活函数，使用ReLU。
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = F.relu

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """
        # 门控（gate）：通过self.gate[layer](x)得到一个用于控制每层输出的权重，通过Sigmoid函数映射到[0, 1]之间。
        # 非线性变换（nonlinear）：通过self.nonlinear[layer](x)执行线性变换，再经过ReLU激活函数。
        # 线性变换（linear）：直接执行线性变换。
        # 加权和：使用门控输出对非线性变换和线性变换进行加权求和
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class GCN(nn.Module):
    # nfeat：输入特征的维度，即每个节点的特征数。
    # nhid：隐藏层特征的维度。
    # nclass：输出类别的数量，对于回归任务，通常为1（如果使用Softmax或sigmoid激活的话，分类任务会用到多类别输出）。
    # dropout：Dropout层的比率，用于防止过拟合。
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN, self).__init__()
        # gc1：第一个图卷积层，输入特征维度为nfeat，输出特征维度为nhid。
        # gc2：第二个图卷积层，输入特征维度为nhid，输出特征维度为nclass
        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    # 实现一些路径上的逐层非线性变换
    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))
    
    # gc1：首先通过第一个图卷积层gc1对输入数据x和图的邻接矩阵adj进行计算，得到隐藏层的表示。应用relu激活函数后，经过dropout，然后输出为隐藏层的表示。
    # gc2：第二个图卷积层gc2使用隐藏层的表示x和邻接矩阵adj进行计算，最终输出结果，即分类或回归任务的预测结果。
    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)#F.dropout 的 training 参数接受一个布尔值：如果 self.training 为 True，则启用 Dropout。如果 self.training 为 False，则不进行 Dropout，直接传递信息。
        x = self.gc2(x, adj)
        return x 
        # return F.log_softmax(x, dim=1)

class GCN_drop_in(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, init):
        super(GCN_drop_in, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, init=init)
        self.gc2 = GraphConvolution(nhid, nclass, init=init)
        self.dropout = dropout

    def bottleneck(self, path1, path2, path3, adj, in_x):
        return F.relu(path3(F.relu(path2(F.relu(path1(in_x, adj)), adj)), adj))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)#先dropout
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)#对第二层图卷积的输出应用 log_softmax 函数，输出对数概率分布，适用于多分类任务

class GAT(nn.Module):
    # nfeat：输入特征的维度（每个节点的特征大小）。
    # nhid：隐藏层特征的维度（每个注意力头输出的特征大小）。
    # nclass：输出类别的数量（用于分类任务）。
    # dropout：Dropout 的概率，用于防止过拟合。
    # alpha：LeakyReLU 激活函数的负半轴斜率。
    # nheads：注意力头的数量，多头注意力机制增强模型的表达能力。
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        # 创建了 nheads 个 GraphAttention 实例，每个实例都是一个独立的注意力头。
        # concat=True：意味着每个注意力头的输出将被拼接起来，从而增强特征表示。
        # 使用 self.add_module 将每个注意力头注册为模块的一部分，以便 PyTorch 正确识别和管理这些子模块。
        self.attentions = [GraphAttention(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 这是一个单独的 GraphAttention 层，用于将多头注意力的输出整合到最终的输出类别中。
        # concat=False：意味着不进行拼接，而是将多头注意力的输出进行平均（或其他整合方式），以得到最终的分类输出。
        self.out_att = GraphAttention(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        # 将输入特征 x 和邻接矩阵 adj 传递给每个注意力头。
        # 将所有注意力头的输出在特征维度上进行拼接（concatenation），增强特征表示。
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)




if __name__ == "__main__":
    gnn = GCN(
            nfeat = 20,
            nhid = 30,
            nclass = 1,
            dropout = 0.6,
            init = 'uniform') #权重初始化方法为均匀初始化。这影响模型训练的初始权重分布，选择合适的初始化方法可以帮助模型更快收敛并避免梯度消失或爆炸的问题。









