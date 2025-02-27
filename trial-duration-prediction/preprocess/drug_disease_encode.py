import csv, pickle
import pandas as pd
from functools import reduce
from tqdm import tqdm
import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import os
import gc

def clean_text(text):
    if pd.isnull(text):
        return None
    # if type(text) != str:
    #     text = ' '.join(text)
    text = text.lower()
    text_split = text.split('\n')
    filter_out_empty_fn = lambda x: len(x.strip())>0
    strip_fn = lambda x:x.strip()
    text_split = list(filter(filter_out_empty_fn, text))
    text_split = list(map(strip_fn, text))
    return ''.join(text_split)

def get_all_texts(feature):#返回所有text列的非重text
    input_file = '../data/raw_data.csv'
    raw_data = pd.read_csv(input_file)
    raw_data[feature] = raw_data[feature].apply(lambda x: ';'.join(eval(x)))
    raw_data = raw_data[[feature]]
    texts = []
    for idx, row in raw_data.iterrows():
        texts.append(clean_text(row[feature]))
        # if row[feature] == "rhumab vegf (telbermin)":
        #     print("flag0")
        #     print(len(row[feature]), row[feature])
        #     print(len(clean_text(row[feature])), row[feature])
    # print(texts)

    return set(texts)

def save_sentence_bert_dict_pkl(feature, model_name):
    cleaned_sentence_set = get_all_texts(feature)

    print(f"save {feature}2embedding.pkl")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def text2vec(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        # 使用 [CLS] token 的输出作为句子的表示
        sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()
        return sentence_vector

    # 创建字典以存储句子和其对应的嵌入向量
    sentence_2_embedding = {}

    for sentences in tqdm(cleaned_sentence_set, desc="Processing sentences", unit="sentence"):
        if len(sentences) == 0:
            print(f"Warning: Empty {feature} list is found")
            sentences_emb = torch.zeros(768, dtype=torch.float32)
        else:
            sentences_emb = torch.mean(torch.stack([text2vec(sentence) for sentence in sentences.split(';')]), dim=0)
        sentence_2_embedding[sentences] = sentences_emb


    output_dir = '../data'

    output_file = os.path.join(output_dir, f'{feature}2embedding.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(sentence_2_embedding, f)

    print(f"Embedding saved to {output_file}")
    return

def load_feature_2_vec(feature):
    sentence_2_vec = pickle.load(open(f'../data/{feature}2embedding.pkl', 'rb'))
    return sentence_2_vec

def feature2embedding(feature, text_lst):
    sentence_2_vec = load_feature_2_vec(feature)
    # text_feature = [torch.tensor(sentence_2_vec[sentence]).view(1,-1) for sentence in text_lst if sentence in sentence_2_vec]
    # if text_feature == []:
    #     text_feature = torch.zeros(1,768)
    # else:
    #     text_feature = torch.cat(text_feature, 0)
    # return text_feature# 4, 768/1, 768
    text_emb = []

    for texts in text_lst:
        texts = clean_text(texts)
        if not texts:
            print(f"Warning: Empty {feature} is found")
            text_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            try:
                text_emb.append(sentence_2_vec[texts])
            except:
                print(f"Warning: Error {feature}")
                text_emb.append(torch.zeros(768, dtype=torch.float32))

    return torch.stack(text_emb)  # len(text_list), 768

if __name__ == "__main__":
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    save_sentence_bert_dict_pkl('drugs', model_name)
    save_sentence_bert_dict_pkl('diseases', model_name)
