'''
input:
	data/raw_data.csv

output: 
	data/icdcode2ancestor_dict.pkl (icdcode to its ancestors)
	icdcode_embedding 

'''

import csv, re, pickle, os 
from functools import reduce 
import icd10
from collections import defaultdict
import numpy as np
import pandas as pd
import torch 
torch.manual_seed(0)
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data  #### data.Dataset 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def text_2_lst_of_lst(text):
	"""
		"[""['F53.0', 'P91.4', 'Z13.31', 'Z13.32']""]"
	"""
	text = text[2:-2]
	code_sublst = []
	for i in text.split('", "'):
		i = i[1:-1]
		code_sublst.append([j.strip()[1:-1] for j in i.split(',')])
	# print(code_sublst)	
	return code_sublst #['F53.0', 'P91.4', 'Z13.31', 'Z13.32']

def get_icdcode_lst():
	input_file = '../data/raw_data.csv'
	with open(input_file, 'r',encoding='utf-8') as csvfile:
		rows = list(csv.reader(csvfile, delimiter = ','))[1:]
	code_lst = []
	for row in rows:
		code_sublst = text_2_lst_of_lst(row[6])
		code_lst.append(code_sublst)
	return code_lst #[['F53.0', 'P91.4', 'Z13.31', 'Z13.32'],...]

def combine_lst_of_lst(lst_of_lst):
	lst = list(reduce(lambda x,y:x+y, lst_of_lst))
	lst = list(set(lst))
	return lst #将嵌套列表去重扁平化

def collect_all_icdcodes():
	code_lst = get_icdcode_lst()
	code_lst = list(map(combine_lst_of_lst, code_lst))
	code_lst = list(reduce(lambda x,y:x+y, code_lst))
	code_lst = list(set(code_lst))
	return code_lst

def find_ancestor_for_icdcode(icdcode, icdcode2ancestor):
	if icdcode in icdcode2ancestor:
		return 
	icdcode2ancestor[icdcode] = []
	ancestor = icdcode[:]
	while len(ancestor) > 2:
		ancestor = ancestor[:-1]
		if ancestor[-1]=='.':
			ancestor = ancestor[:-1]
		if icd10.find(ancestor) is not None:
			icdcode2ancestor[icdcode].append(ancestor)
	return

def build_icdcode2ancestor_dict():#生成icdcode祖宗集pkl文件，{icdcode:[ancestor1,ancestor2,..],..}
	pkl_file = "../data/icdcode2ancestor_dict.pkl"
	if os.path.exists(pkl_file):
		icdcode2ancestor = pickle.load(open(pkl_file, 'rb'))
		return icdcode2ancestor 
	all_code = collect_all_icdcodes() 
	icdcode2ancestor = defaultdict(list)
	for code in all_code:
		find_ancestor_for_icdcode(code, icdcode2ancestor)
	pickle.dump(icdcode2ancestor, open(pkl_file,'wb'))
	return icdcode2ancestor 

def collect_all_code_and_ancestor():#返回icdcode和其祖宗的全部不重复集合
	icdcode2ancestor = build_icdcode2ancestor_dict()
	all_code = set(icdcode2ancestor.keys())
	ancestor_lst = list(icdcode2ancestor.values())
	ancestor_set = set(reduce(lambda x,y:x+y, ancestor_lst))
	all_code_lst = all_code.union(ancestor_set)
	return all_code_lst	

class GRAM(nn.Sequential):
	"""	
		return a weighted embedding 
	"""

	def __init__(self, embedding_dim, icdcode2ancestor, device):
		super(GRAM, self).__init__()		
		self.icdcode2ancestor = icdcode2ancestor 
		self.all_code_lst = GRAM.codedict_2_allcode(self.icdcode2ancestor)
		self.code_num = len(self.all_code_lst)
		self.maxlength = 5
		self.code2index = {code:idx for idx,code in enumerate(self.all_code_lst)}
		self.index2code = {idx:code for idx,code in enumerate(self.all_code_lst)}
		self.padding_matrix = torch.zeros(self.code_num, self.maxlength).long() #padding_matrix 用来存储代码和其祖先代码的索引。
		self.mask_matrix = torch.zeros(self.code_num, self.maxlength)#mask_matrix 用来存储代码和祖先的掩码值（1 表示有效，0 表示无效）
		for idx in range(self.code_num):
			code = self.index2code[idx]
			ancestor_code_lst = self.icdcode2ancestor[code]
			ancestor_idx_lst = [idx] + [self.code2index[code] for code in ancestor_code_lst]
			ancestor_mask_lst = [1 for i in ancestor_idx_lst] + [0] * (self.maxlength - len(ancestor_idx_lst))
			ancestor_idx_lst = ancestor_idx_lst + [0]*(self.maxlength-len(ancestor_idx_lst))
			self.padding_matrix[idx,:] = torch.Tensor(ancestor_idx_lst)
			self.mask_matrix[idx,:] = torch.Tensor(ancestor_mask_lst)

		self.embedding_dim = embedding_dim 
		self.embedding = nn.Embedding(self.code_num, self.embedding_dim)
		self.attention_model = nn.Linear(2*embedding_dim, 1)

		self.device = device
		self = self.to(device)
		self.padding_matrix = self.padding_matrix.to('cpu')
		self.mask_matrix = self.mask_matrix.to('cpu')

	@property
	def embedding_size(self):
		return self.embedding_dim


	@staticmethod
	def codedict_2_allcode(icdcode2ancestor):
		all_code = set(icdcode2ancestor.keys())
		ancestor_lst = list(icdcode2ancestor.values())
		ancestor_set = set(reduce(lambda x,y:x+y, ancestor_lst))
		all_code_lst = all_code.union(ancestor_set)
		return all_code_lst		


	def forward_single_code(self, single_code):
		idx = self.code2index[single_code].to(self.device)
		ancestor_vec = self.padding_matrix[idx,:]  #### (5,)
		mask_vec = self.mask_matrix[idx,:] 

		embeded = self.embedding(ancestor_vec)  ### 5, 50
		current_vec = torch.cat([self.embedding(torch.Tensor([idx]).long()).view(1,-1) for i in range(self.maxlength)], 0) ### 1,50 -> 5,50
		attention_input = torch.cat([embeded, current_vec], 1)  ### 5, 100
		attention_weight = self.attention_model(attention_input)  ##### 5, 1
		attention_weight = torch.exp(attention_weight)  #### 5, 1
		attention_output = attention_weight * mask_vec.view(-1,1)  #### 5, 1
		attention_output = attention_output / torch.sum(attention_output)  #### 5, 1
		output = embeded * attention_output ### 5, 50 
		output = torch.sum(output, 0) ### 50
		return output 


	def forward_code_lst(self, code_lst):
		"""
			
			['C05.2', 'C10.0', 'C16.0', 'C16.4', 'C17.0', 'C17.1', 'C17.2'], length is 32 
			32 is length of code_lst; 5 is maxlength; 50 is embedding_dim; 
		"""
		idx_lst = [self.code2index[code] for code in code_lst if code in self.code2index] ### 32 
		if idx_lst == []:
			idx_lst = [0]
		ancestor_mat = self.padding_matrix[idx_lst,:].to(self.device)  ##### 32,5
		mask_mat = self.mask_matrix[idx_lst,:].to(self.device)  #### 32,5
		embeded = self.embedding(ancestor_mat)  #### 32,5,50
		current_vec = self.embedding(torch.Tensor(idx_lst).long().to(self.device)) #### 32,50
		current_vec = current_vec.unsqueeze(1) ### 32,1,50
		current_vec = current_vec.repeat(1, self.maxlength, 1) #### 32,5,50
		attention_input = torch.cat([embeded, current_vec], 2)  #### 32,5,100
		attention_weight = self.attention_model(attention_input)  #### 32,5,1 
		attention_weight = torch.exp(attention_weight).squeeze(-1)  #### 32,5
		attention_output = attention_weight * mask_mat  #### 32,5 
		attention_output = attention_output / torch.sum(attention_output, 1).view(-1,1)  #### 32,5 
		attention_output = attention_output.unsqueeze(-1)  #### 32,5,1 
		output = embeded * attention_output ##### 32,5,50 
		output = torch.sum(output,1) ##### 32,50
		return output 

	def forward_code_lst2(self, code_lst_lst):
		### in one sample 
		code_lst = reduce(lambda x,y:x+y, code_lst_lst)
		code_embed = self.forward_code_lst(code_lst)
		### to do 
		code_embed = torch.mean(code_embed, 0).view(1,-1)  #### dim, 
		return code_embed #(1,dim)
		
	def forward_code_lst3(self, code_lst_lst_lst):
		code_embed_lst = [self.forward_code_lst2(code_lst_lst) for code_lst_lst in code_lst_lst_lst]
		code_embed = torch.cat(code_embed_lst, 0)
		return code_embed #(len,dim)

def query_ancestor(icdcode, icdcode2ancestor_dict):
    if icdcode in icdcode2ancestor_dict:
        return icdcode2ancestor_dict[icdcode]
    else:
        return None

#生成icdcodes的嵌入向量
def generate_embeddings(data):
	pkl_path = '../data/icdcode2ancestor_dict.pkl'
	output_pkl_path = '../data/icdcode2ancestor_embedding_dict.pkl'
	if os.path.exists(output_pkl_path):
		with open(output_pkl_path, 'rb') as f:
			icdcodes_embeddings = pickle.load(f)
			return icdcodes_embeddings
	with open(pkl_path, 'rb') as f:
		icdcode2ancestor_dict = pickle.load(f)
	gram_model = GRAM(embedding_dim=50,icdcode2ancestor=icdcode2ancestor_dict,device='cpu')
	triple_list = []
	valid_icdcodes = []
	for row in data:
		if row.strip() == '[]': #检查空的icdcodes
			triple_list.append([]) #空的icdcodes转为空列表
			valid_icdcodes.append('') #标记为空
		else:
			triple_list.append(text_2_lst_of_lst(row))  #转换为嵌套列表
			valid_icdcodes.append(row)  #保存有效icdcodes
	embeddings = gram_model.forward_code_lst3(triple_list).detach().numpy()
	icdcodes_embeddings = {}
	for i, code in enumerate(valid_icdcodes):
		if code != '[]':
			if code not in icdcodes_embeddings:
				icdcodes_embeddings[code] = embeddings[i]  #第一次遇到该icdcode生成嵌入向量
		elif code not in icdcodes_embeddings: #code为空且未保存过
			icdcodes_embeddings[code] = np.zeros(50, dtype=np.float32)  #空的icdcodes赋值为全零向量
	with open(output_pkl_path, 'wb') as f:
		pickle.dump(icdcodes_embeddings, f)
	return icdcodes_embeddings



def query_embedding_icdcodes(icd_lst):
	output_pkl_path = '../data/icdcode2ancestor_embedding_dict.pkl'
	with open(output_pkl_path, 'rb') as f:
		embeddings_dict = pickle.load(f)
	emb_lst = []
	for icdcodes in icd_lst:
		try:
			emb = embeddings_dict[icdcodes]
		except KeyError:
			emb = np.zeros(50, dtype=np.float32)
		emb_lst.append(emb)
	return emb_lst
	# if icdcodes in embeddings_dict:
	# 	return embeddings_dict[icdcodes]  #返回对应嵌入向量
	# else:
	# 	return np.zeros(50, dtype=np.float32)


if __name__ == '__main__':
	dic = build_icdcode2ancestor_dict()
	df = pd.read_csv('../data/raw_data.csv',sep=',',dtype={'masking': str, 'intervention_model': str})
	data = df['icdcodes'].apply(lambda x: ';'.join(eval(x))).tolist()
	generate_embeddings(data)


