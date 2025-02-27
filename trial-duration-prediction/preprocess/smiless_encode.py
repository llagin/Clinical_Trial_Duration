from preprocess.molecule_encode import MPNN,smiles2mpnnfeature,txt_to_lst
import torch
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(pkl_file):
    with open(pkl_file, 'rb') as f:
        smiles_embeddings = pickle.load(f)
    return smiles_embeddings

#查询指定 SMILES 列表对应的嵌入向量。
def query_embeddings(smiles_lst_lst, embedding_dim=50):
    embeddings_dict = load_model('../data/smiless2embedding.pkl')
    emb_lst = []
    for smiles_list in smiles_lst_lst:
        smiles_list = txt_to_lst(smiles_list)

        #smiles_list为空，返回全零向量
        if not smiles_list:
            emb = np.zeros(embedding_dim, dtype=np.float32)
            emb_lst.append(emb)
            continue
        #查询嵌入向量
        valid_embeddings = []
        for smiles in smiles_list:
            key = tuple([smiles])  #将单个SMILES转为元组形式以匹配字典
            if key in embeddings_dict:
                valid_embeddings.append(embeddings_dict[key])

        #未找到任何有效嵌入，返回全零向量
        if not valid_embeddings:
            emb = np.zeros(embedding_dim, dtype=np.float32)
            emb_lst.append(emb)
            continue

        #计算有效嵌入向量的平均值
        valid_embeddings = np.stack(valid_embeddings).astype(np.float32)  #转为NumPy数组
        mean_embedding = np.mean(valid_embeddings, axis=0)  #计算平均值
        emb_lst.append(mean_embedding)
    # mean_embedding = np.sum(valid_embeddings, axis=0)  #计算和               #这里没测试是加和好还是平均好
    return emb_lst

def generate_embeddings(smiless_lst_lst, output_pkl, hidden_size=50, depth=3, device='cpu'):
    if os.path.exists(output_pkl):
        return
    
    model = MPNN(mpnn_hidden_size=hidden_size, mpnn_depth=depth, device=torch.device(device))
    try:
        embeddings = model.forward_smiles_lst_lst(smiless_lst_lst)
    except Exception as e:
        print("嵌入出错")
        return
    
    #构建字典：输入SMILES列表->嵌入向量
    smiles_embeddings = {tuple(s_lst): embeddings[i].detach().cpu().numpy() for i, s_lst in enumerate(smiless_lst_lst)}
    with open(output_pkl, 'wb') as f:
        pickle.dump(smiles_embeddings, f)



if __name__ == "__main__":
    #构建smiles模型
    df = pd.read_csv('../data/raw_data.csv', sep=',')
    df['smiless'] = df['smiless'].fillna('')
    s_lst_lst = df['smiless'].tolist()
    smiless_lst_lst = []
    for data in s_lst_lst:
        try:
            s_lst = txt_to_lst(data)
            smiless_lst_lst.append(s_lst)
        except Exception as e:
            smiless_lst_lst.append([])
    smiless_lst_lst = [lst for lst in smiless_lst_lst if lst]
    output_pkl = '../data/smiless2embedding.pkl'
    generate_embeddings(smiless_lst_lst,output_pkl)

    #嵌入查询测试
    smiles_to_query = ['[H][C@@]12CC[C@H](C(C)=O)[C@@]1(C)CC[C@]1([H])[C@@]2([H])C=CC2=CC(=O)CC[C@@]12C']
    embedding = query_embeddings(smiles_to_query)
    print(embedding)
    print(embedding.shape)
