'''
input: 
	smiles batch
 


utility
	1. graph MPN
	2. smiles 
	3. morgan feature 

output:
	1. embedding batch 



deeppurpose
	DDI
	encoders  model 

to do 
	lst -> dataloader -> feature -> model 


	mpnn's feature -> collate -> model 

'''

import csv 
from tqdm import tqdm 
import numpy as np
from copy import deepcopy 
import matplotlib.pyplot as plt
import ast
import rdkit
import rdkit.Chem as Chem 
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.*')  
# 对手性碳的处理
# from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import torch 
torch.manual_seed(0)
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data  #### data.Dataset 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from preprocess.module import Highway

def get_drugbank_smiles_lst():
	drugfile = 'data/drugbank_drugsmiles.csv'
	with open(drugfile, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter='\t', skipinitialspace=True)
		header = next(reader)
		try:
			smiles_idx = header.index('smiles')
		except ValueError:
			raise ValueError("CSV 文件中未找到 'smiles' 列")
		smiles_lst = [row[smiles_idx] for row in reader]
	return smiles_lst

# def txt_to_lst(text):
# 	"""
# 		"['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
# 	"""
# 	text = text[1:-1]
# 	lst = [i.strip()[1:-1] for i in text.split(' ')]
# 	return lst 
def txt_to_lst(text):
    lst = ast.literal_eval(text)
    if isinstance(lst, list):  # 确保解析结果为列表
        return lst
    else:
        raise []

# def get_cooked_data_smiles_lst():# 获得所有去重的smiles
# 	cooked_file = '../data/drugbank_drugsmiles.csv'
# 	with open(cooked_file, 'r', encoding="utf-8") as csvfile:
# 		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
# 	smiles_lst = [row[56] for row in rows]
# 	print(rows[0][56])
# 	smiles_lst = list(map(txt_to_lst, smiles_lst))
# 	from functools import reduce
# 	smiles_lst = list(reduce(lambda x,y:x+y, smiles_lst))
# 	smiles_lst = list(set(smiles_lst))
# 	# print(len(smiles_lst))  
# 	return smiles_lst


# 将一个张量转换为 PyTorch 的 Variable，并根据需要设置 requires_grad 属性，后面可能会删除这个函数
def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)

# 在指定维度上对高维张量进行索引选择，同时保持张量的多维结构。
# source：源张量，形状为 (D0, D1, D2, ..., Dn)。
# dim：需要进行索引选择的维度。
# index：索引张量，通常是一维的，包含要选择的索引值。
def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def get_mol(smiles): #返回Kekul化后的SMILES字符串，例如苯就变成了 'C1=CC=CC=C1'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

# 每个原子特征向量的维度，总共 39 维。
# 23 维：元素类型独热编码。
# 6 维：原子度数独热编码。
# 5 维：原子形式电荷独热编码。
# 4 维：手性标签独热编码。
# 1 维：芳香性二元特征
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1

# 每个化学键特征向量的维度，总共 11 维。
# 5 维：键类型独热编码。
# 6 维：键立体化学独热编码。
BOND_FDIM = 5 + 6

# 每个原子的最大邻居数量，设定为 6。
# 处理邻居数量不足的原子，通过零填充保持一致性。
# 处理邻居数量超过的原子，仅保留前 MAX_NB 个邻居。
# 用于构建原子和键的邻接关系张量，确保模型输入的一致性。
MAX_NB = 6
### basic setting from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/mpn.py

#将一个值 x 转换为独热编码。如果 x 不在 allowable_set 中，则使用 allowable_set 的最后一个元素
def onek_encoding_unk(x, allowable_set):#独热编码
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))#一个独热编码的列表，长度等于 allowable_set 的长度
# 实例
# onek_encoding_unk('X', ['C', 'N', 'O', 'unknown'])
# 输出: [False, False, False, True]


# 生成一个原子的特征向量，包含多个独热编码和二元特征。
def atom_features(atom):#返回张量
	# 元素类型：通过 ELEM_LIST 进行独热编码。
	# 原子度数（Degree）：通过 [0,1,2,3,4,5] 进行独热编码。
	# 形式电荷（Formal Charge）：通过 [-1,-2,1,2,0] 进行独热编码。
	# 手性标签（Chiral Tag）：通过 [0,1,2,3] 进行独热编码。
	# 芳香性（Is Aromatic）：单一二元特征，表示原子是否为芳香环的一部分。
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])

# 生成一个化学键的特征向量，包含键类型的独热编码和键立体化学信息的独热编码
def bond_features(bond):#返回张量
	# 键类型（Bond Type）：通过 [单键, 双键, 三键, 芳香键, 环键] 进行独热编码。
	# 键立体化学（Stereo）：通过 [0,1,2,3,4,5] 进行独热编码，表示键的立体化学信息。
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo) # 一个包含所有特征的张量，长度为 BOND_FDIM

# 将 SMILES 字符串转换为适用于消息传递神经网络（Message Passing Neural Networks, MPNN）模型的特征表示。
def smiles2mpnnfeature(smiles):
	## from mpn.py::tensorize  
	'''
		data-flow:   
			data_process(): apply(smiles2mpnnfeature)
			DBTA: train(): data.DataLoader(data_process_loader())
			mpnn_collate_func()


		padding: 一个零向量，用于在后续处理中填充空的键（bond）。
		fatoms: 存储所有原子的特征列表。
		fbonds: 存储所有键（bond）的特征列表，初始化时包含一个 padding 向量。
		in_bonds: 每个原子的入边列表。
		all_bonds: 存储所有的键对（bond pairs），初始化时包含一个无效的键对 (-1, -1)。
	'''
	# padding：一个零向量，用于填充键特征，以处理节点邻居数量不足的情况。
	# fatoms：存储所有原子的特征列表。
	# fbonds：存储所有键的特征列表，初始化时包含一个 padding 向量，用于后续的索引对齐。
	# in_bonds：每个原子的入边列表，记录每个原子所连接的键的索引。
	# all_bonds：存储所有键对（bond pairs），初始化时包含一个无效的键对 (-1, -1)，用于占位。
	padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
	fatoms, fbonds = [], [padding] 
	in_bonds,all_bonds = [], [(-1,-1)] 

	# 将 SMILES 字符串转换为 RDKit 的 Mol 对象，并进行 Kekulize 处理。如果 SMILES 无效，返回 None。
	mol = get_mol(smiles)

	# atom_features(atom)：调用之前定义的 atom_features 函数，生成一个原子的特征向量，并将其添加到 fatoms 列表中。
	# in_bonds.append([])：初始化每个原子的入边列表，后续用于记录该原子连接的键的索引。
	if mol is not None:
		n_atoms = mol.GetNumAtoms()
		for atom in mol.GetAtoms():
			fatoms.append(atom_features(atom))
			in_bonds.append([])

		for bond in mol.GetBonds():
			a1 = bond.GetBeginAtom()
			a2 = bond.GetEndAtom()
			x = a1.GetIdx() 
			y = a2.GetIdx()

			b = len(all_bonds)
			all_bonds.append((x,y))
			fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
			in_bonds[y].append(b)# b指向all_bond对应的一条边

			b = len(all_bonds)
			all_bonds.append((y,x))
			fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
			in_bonds[x].append(b)

		total_bonds = len(all_bonds)#计算总键数，包括对称键。
		fatoms = torch.stack(fatoms, 0)#将原子特征列表堆叠成二维张量，形状 (n_atoms, ATOM_FDIM)
		fbonds = torch.stack(fbonds, 0)#将键特征列表堆叠成二维张量，形状 (total_bonds, BOND_FDIM)
		agraph = torch.zeros(n_atoms,MAX_NB).long()#原子之间的邻接关系
		bgraph = torch.zeros(total_bonds,MAX_NB).long()#键之间的邻接关系

		#为每个原子填充其入边的键索引。
		# 遍历每个原子 a。
		# 遍历该原子的入边 in_bonds[a]，将键索引 b 填入 agraph[a, i]。
		# 如果某个原子的入边数量小于 MAX_NB，剩余位置保持为零（padding）。
		for a in range(n_atoms):
			for i,b in enumerate(in_bonds[a]):
				agraph[a,i] = b

		#为每个键填充其邻接键的索引。
		# 从 b1 = 1 开始遍历所有键（跳过第一个无效键对 (-1, -1)）。
		# 获取当前键的起始和结束原子 x 和 y。
		# 遍历起始原子 x 的入边 in_bonds[x]，获取邻接键 b2。
		# 条件判断：仅当 all_bonds[b2][0] != y 时，才将 b2 填入 bgraph[b1, i]，避免与当前键形成环路。
		# 如果某个键的邻接键数量小于 MAX_NB，剩余位置保持为零（padding）。
		for b1 in range(1, total_bonds):
			x,y = all_bonds[b1]
			for i,b2 in enumerate(in_bonds[x]):
				if all_bonds[b2][0] != y:
					bgraph[b1,i] = b2

	#当 SMILES 无效或无法解析时，返回空的特征张量
	else: 
		# print('Molecules not found and change to zero vectors..')
		fatoms = torch.zeros(0,39)
		fbonds = torch.zeros(0,50)
		agraph = torch.zeros(0,6)
		bgraph = torch.zeros(0,6)

	# Natom 和 Nbond：分别为原子数和键数。
	# shape_tensor：存储分子的原子数和键数，形状为 (1, 2)。
	# 返回值：一个包含五个二维张量的列表：
	# fatoms.float()：原子特征张量，形状 (Natom, ATOM_FDIM)。
	# fbonds.float()：键特征张量，形状 (Nbond, BOND_FDIM)。
	# agraph.float()：原子邻接关系张量，形状 (Natom, MAX_NB)。
	# bgraph.float()：键邻接关系张量，形状 (Nbond, MAX_NB)。
	# shape_tensor：分子形状信息张量，形状 (1, 2)。
	Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
	shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
	return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor]#5个二维张量


class smiles_dataset(data.Dataset):
	#存储 SMILES 列表和标签列表
	def __init__(self, smiles_lst, label_lst):
		self.smiles_lst = smiles_lst 
		self.label_lst = label_lst 

	def __len__(self):
		return len(self.smiles_lst)
	
	# 根据索引获取对应的 SMILES 字符串和标签。
	# 调用 smiles2mpnnfeature(smiles) 函数将 SMILES 转换为 MPNN 所需的特征表示。
	# 返回特征表示和标签。
	def __getitem__(self, index):
		smiles = self.smiles_lst[index]
		label = self.label_lst[index]
		smiles_feature = smiles2mpnnfeature(smiles)
		return smiles_feature, label 

## DTI.py --> collate 

## x is a list, len(x)=batch_size, x[i] is tuple, len(x[0])=5  

#将单个样本的特征合并成批量数据。
def mpnn_feature_collate_func(x): 
	return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]

#在数据加载时将批次中的样本特征和标签进行整理和拼接，以便输入到 MPNN 模型中
def mpnn_collate_func(x):
	#print("len(x) is ", len(x)) ## batch_size 
	#print("len(x[0]) is ", len(x[0])) ## 3--- data_process_loader.__getitem__ 
	mpnn_feature = [i[0] for i in x]#list of list
	#print("len(mpnn_feature)", len(mpnn_feature), "len(mpnn_feature[0])", len(mpnn_feature[0]))
	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
	#提取每个样本的剩余特征（除了 MPNN 特征外的其他部分），通常是标签
	from torch.utils.data.dataloader import default_collate
	x_remain = [i[1:] for i in x]
	x_remain_collated = default_collate(x_remain)
	#将拼接后的 MPNN 特征与批量的剩余特征（如标签）组合，形成最终的批量输出
	return [mpnn_feature] + x_remain_collated


def data_loader():
	smiles_lst = get_drugbank_smiles_lst() # 这里修改了，原来用的是去重之后的smiles的list，但是在这里的数据没有重复数据
	label_lst = [1 for i in range(len(smiles_lst))]	
	dataset = smiles_dataset(smiles_lst, label_lst)
	dataloader = data.DataLoader(dataset, batch_size=32, collate_fn = mpnn_collate_func, ) 
	return dataloader 


class MPNN(nn.Sequential):# smils_graph编码
	# mpnn_hidden_size：隐藏层的维度。
	# mpnn_depth：消息传递的深度，即消息传递层的数量。
	def __init__(self, mpnn_hidden_size, mpnn_depth, device):
		super(MPNN, self).__init__()
		self.mpnn_hidden_size = mpnn_hidden_size
		self.mpnn_depth = mpnn_depth 
		# self.W_i：用于处理键特征的线性层，将 ATOM_FDIM + BOND_FDIM 维的输入映射到隐藏维度。
		# self.W_h：用于聚合邻居消息的线性层，将隐藏维度映射回隐藏维度。
		# self.W_o：用于输出层，将 ATOM_FDIM + mpnn_hidden_size 维的输入映射到隐藏维度。
		self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
		self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

		self.device = device
		self = self.to(self.device)

	def set_device(self, device):
		self.device = device 

	#返回嵌入的维度，即隐藏层的大小。
	@property
	def embedding_size(self):
		return self.mpnn_hidden_size 

	### forward single molecule sequentially. 
	#处理单个批次（通常为单个分子）的特征，将其转换为嵌入向量。
	def feature_forward(self, feature):
		''' 
			batch_size == 1 
			feature: utils.smiles2mpnnfeature 
		'''
		fatoms, fbonds, agraph, bgraph, atoms_bonds = feature #feature：来自 smiles2mpnnfeature 函数的五元组，包括 fatoms、fbonds、agraph、bgraph、atoms_bonds。
		agraph = agraph.long()
		bgraph = bgraph.long()
		#print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape, atoms_bonds.shape)
		atoms_bonds = atoms_bonds.long()
		batch_size = atoms_bonds.shape[0]

		# N_atoms 和 N_bonds 用于跟踪当前处理的原子和键的索引。
		# embeddings 列表用于存储每个分子的嵌入向量。
		N_atoms, N_bonds = 0, 0 
		embeddings = []

		for i in range(batch_size):
			n_a = atoms_bonds[i,0].item()
			n_b = atoms_bonds[i,1].item()

			# 提取当前分子的原子数 n_a 和键数 n_b。如果 n_a 为 0，表示无效分子，添加一个零向量嵌入。
			if (n_a == 0):
				embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
				embeddings.append(embed.to(self.device))
				continue 

			# 否则，提取当前分子的原子特征、键特征、原子邻接关系和键邻接关系。
			sub_fatoms = fatoms[N_atoms:N_atoms+n_a,:].to(self.device)
			sub_fbonds = fbonds[N_bonds:N_bonds+n_b,:].to(self.device)
			sub_agraph = agraph[N_atoms:N_atoms+n_a,:].to(self.device)
			sub_bgraph = bgraph[N_bonds:N_bonds+n_b,:].to(self.device)

			# 调用 single_feature_forward 方法计算嵌入向量，并将其添加到 embeddings 列表。
			embed = self.single_feature_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
			embed = embed.to(self.device)           
			embeddings.append(embed)

			# 更新原子和键的索引计数器。
			N_atoms += n_a
			N_bonds += n_b
		if len(embeddings)==0:
			return None 
		else:
			return torch.cat(embeddings, 0)
		
	#执行消息传递操作，将原子和键特征转换为分子的嵌入向量。
	# fatoms：原子特征张量，形状 (x, 39)。
	# fbonds：键特征张量，形状 (y, 50)。
	# agraph：原子邻接关系张量，形状 (x, 6)。
	# bgraph：键邻接关系张量，形状 (y, 6)。
	def single_feature_forward(self, fatoms, fbonds, agraph, bgraph):
		'''
			fatoms: (x, 39)
			fbonds: (y, 50)
			agraph: (x, 6)
			bgraph: (y,6)
		'''
		### invalid molecule
		if fatoms.shape[0] == 0:
			return create_var(torch.zeros(1, self.mpnn_hidden_size).to(self.device))
		agraph = agraph.long()
		bgraph = bgraph.long()
		fatoms = create_var(fatoms).to(self.device)
		fbonds = create_var(fbonds).to(self.device)
		agraph = create_var(agraph).to(self.device)
		bgraph = create_var(bgraph).to(self.device)

		#将键特征通过线性层 W_i 转换为隐藏维度。
		binput = self.W_i(fbonds)
		message = F.relu(binput)
		#print("shapes", fbonds.shape, binput.shape, message.shape)
		for i in range(self.mpnn_depth - 1):
			#选择与每个键相邻的消息。
			nei_message = index_select_ND(message, 0, bgraph)

			#聚合邻居消息。
			nei_message = nei_message.sum(dim=1)

			#通过线性层 W_h 转换聚合后的消息。
			nei_message = self.W_h(nei_message)

			#更新消息。
			message = F.relu(binput + nei_message)

		#选择与每个原子相邻的消息。
		nei_message = index_select_ND(message, 0, agraph)

		#聚合邻居消息。
		nei_message = nei_message.sum(dim=1)

		#连接原子特征和聚合消息。
		ainput = torch.cat([fatoms, nei_message], dim=1)

		#通过线性层 W_o 转换，并应用 ReLU 激活函数。
		atom_hiddens = F.relu(self.W_o(ainput))

		#对所有原子嵌入取均值，生成分子嵌入向量。
		return torch.mean(atom_hiddens, 0).view(1,-1)


	#将单个 SMILES 字符串编码为嵌入向量。
	def forward_single_smiles(self, smiles):
		fatoms, fbonds, agraph, bgraph, _ = smiles2mpnnfeature(smiles)
		embed = self.single_feature_forward(fatoms, fbonds, agraph, bgraph).view(1,-1)
		return embed

	#将 SMILES 列表编码为嵌入向量列表，并将其拼接为一个批量张量。
	def forward_smiles_lst(self, smiles_lst):
		embed_lst = [self.forward_single_smiles(smiles) for smiles in smiles_lst]
		embed_all = torch.cat(embed_lst, 0)
		return embed_all

	#将 SMILES 列表编码为嵌入向量，计算其平均值作为整体嵌入。
	def forward_smiles_lst_average(self, smiles_lst): 
		embed_all = self.forward_smiles_lst(smiles_lst)
		embed_avg = torch.mean(embed_all, 0).view(1,-1)
		return embed_avg

	#将多个 SMILES 列表编码为嵌入向量，每个 SMILES 列表的嵌入取平均后拼接。
	def forward_smiles_lst_lst(self, smiles_lst_lst): 
		embed_lst = [self.forward_smiles_lst_average(smiles_lst) for smiles_lst in smiles_lst_lst]
		embed_all = torch.cat(embed_lst, 0)  #### n,dim
		return embed_all


class ADMET(nn.Sequential):#embed和label进行监督学习

	# molecule_encoder：分子编码器（例如 MPNN 实例）。
	# highway_num：Highway 层的数量。
	# device：模型运行的设备（如 'cuda' 或 'cpu'）。
	# epoch：训练的轮数。
	# lr：学习率。
	# weight_decay：权重衰减（正则化参数）。
	# save_name：模型保存的名称。
	def __init__(self, molecule_encoder, highway_num, device,  
					epoch, lr, weight_decay, save_name):
		super(ADMET, self).__init__()
		# self.highway_nn_lst：包含 5 个 Highway 模块的 ModuleList，每个 Highway 模块的大小为 embedding_size，层数为 highway_num。
		# self.fc_output_lst：包含 5 个线性层的 ModuleList，每个线性层将 embedding_size 维映射到 1 维输出。
		# self.f：激活函数，设置为 F.relu。
		# self.loss：损失函数，设置为二元交叉熵带 logits 的损失函数 BCEWithLogitsLoss。
		self.molecule_encoder = molecule_encoder 
		self.embedding_size = self.molecule_encoder.embedding_size
		self.highway_num = highway_num 
		self.highway_nn_lst = nn.ModuleList([Highway(size = self.embedding_size, num_layers = self.highway_num) for i in range(5)])
		self.fc_output_lst = nn.ModuleList([nn.Linear(self.embedding_size, 1) for i in range(5)])
		self.f = F.relu 
		self.loss = nn.BCEWithLogitsLoss()

		self.epoch = epoch 
		self.lr = lr 
		self.weight_decay = weight_decay 
		self.save_name = save_name 

		self.device = device 
		self = self.to(device)

	def set_device(self, device):
		self.device = device
		self.to(device)
		self.molecule_encoder.set_device(device)

	#将 SMILES 列表编码为嵌入向量，并通过对应的 Highway 模块处理。
	def forward_smiles_lst_embedding(self, smiles_lst, idx):
		embed_all = self.molecule_encoder.forward_smiles_lst(smiles_lst)
		output = self.highway_nn_lst[idx](embed_all)
		return output 

	#将嵌入向量通过对应的全连接层进行预测。
	def forward_embedding_to_pred(self, embeded, idx):
		return self.fc_output_lst[idx](embeded)

	#将 SMILES 列表编码为嵌入向量，并通过对应的线性层进行预测。
	def forward_smiles_lst_pred(self, smiles_lst, idx):
		embeded = self.forward_smiles_lst_embedding(smiles_lst, idx)
		fc_output = self.forward_embedding_to_pred(embeded, idx)
		return fc_output   

	#在验证集上评估模型的性能，计算平均损失。
	def test(self, dataloader_lst, return_loss = True):
		loss_lst = []
		for idx in range(1):
			single_loss_lst = []
			for smiles_lst, label_vec in dataloader_lst[idx]:
				output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
				loss = self.loss(output, label_vec.to(self.device).float())
				single_loss_lst.append(loss.item())
			loss_lst.append(np.mean(single_loss_lst))
		return np.mean(loss_lst)

	#训练模型，使用训练集进行优化，并在每个 epoch 结束后在验证集上评估性能，保存最佳模型。
	def train(self, train_loader_lst, valid_loader_lst):
		#使用 Adam 优化器，设置学习率和权重衰减。
		opt = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

		# 计算初始验证损失并记录。
		# 设置最佳验证损失为初始值，并保存当前模型为最佳模型。
		train_loss_record = [] 
		valid_loss = self.test(valid_loader_lst, return_loss=True)
		valid_loss_record = [valid_loss]
		best_valid_loss = valid_loss 
		best_model = deepcopy(self)
		for ep in tqdm(range(self.epoch)):
			data_iterator_lst = [iter(train_loader_lst[idx]) for idx in range(5)]
			try: 
				while True:
					for idx in range(1):
						smiles_lst, label_vec = next(data_iterator_lst[idx])
						output = self.forward_smiles_lst_pred(smiles_lst, idx).view(-1)
						loss = self.loss(output, label_vec.float()) 
						opt.zero_grad() 
						loss.backward()
						opt.step()	
			except:
				pass 
			valid_loss = self.test(valid_loader_lst, return_loss = True)
			valid_loss_record.append(valid_loss)						

			if valid_loss < best_valid_loss:
				best_valid_loss = valid_loss 
				best_model = deepcopy(self)

		self = deepcopy(best_model)





# if __name__ == "__main__":
# 	model = MPNN(mpnn_hidden_size = 50, mpnn_depth = 3, device = torch.device("cpu"))
# 	dataloader = data_loader()
# 	for smiles_feature, labels in dataloader:
# 		embedding = model(smiles_feature) 
# 		print(embedding.shape)
		












