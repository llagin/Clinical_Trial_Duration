import csv, pickle
import pandas as pd
from functools import reduce
from tqdm import tqdm
import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import os
import json
import gc

wd_2_word = [
    (' yr ', ' year '),
    (' yrs ', ' years '),
    (' mo ', ' month '),
    (' mos ', ' months '),
    (' d ', ' day '),
    (' ds ', ' days '),
    (' wk ', ' week '),
    (' wks ', ' weeks ')
]

def clean_text(text):
    if pd.isnull(text):
        return None
    text = text.lower()
    for k,v in wd_2_word:
        text.replace(k,v)
    text_split = text.split('\n')
    filter_out_empty_fn = lambda x: len(x.strip())>0
    strip_fn = lambda x:x.strip()
    text_split = list(filter(filter_out_empty_fn, text_split))
    text_split = list(map(strip_fn, text_split))
    text = ''.join(text_split)
    return text

def get_all_texts():#返回所有text列的非重text
    input_file = '../data/raw_data.csv'
    raw_data = pd.read_csv(input_file)
    raw_data = raw_data[['time_frame']]
    texts = []
    for idx, row in raw_data.iterrows():
        texts.extend([clean_text(text) for text in eval(row['time_frame'])])
    # print(texts)
    return set(texts)

def save_sentence_bert_dict_pkl(batch_size=64, save_interval=800000):
    cleaned_sentence_set = get_all_texts()

    print("save time_frame2idx")
    text2idx = {sentence: index for index, sentence in enumerate(cleaned_sentence_set)}

    with open('../data/time_frame2id.json', 'w') as json_file:
        json.dump(text2idx, json_file)

    print("save time_frame2embedding")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    sentence_embeddings = []
    batch_sentences = []

    cleaned_sentence_list = list(cleaned_sentence_set)

    ctr = 1

    for i, sentence in enumerate(tqdm(cleaned_sentence_list)):
        batch_sentences.append(sentence)

        if len(batch_sentences) == batch_size or i == len(cleaned_sentence_list) - 1:
            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs)

            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            for emb in cls_embeddings:
                sentence_embeddings.append(emb.tolist())

            batch_sentences = []

        if (i + 1) % save_interval == 0 or i == len(cleaned_sentence_list) - 1:
            sentence_embeddings_tensor = torch.tensor(sentence_embeddings)
            save_path = f"../data/time_frame_emb_{ctr * save_interval}.pt"

            torch.save(sentence_embeddings_tensor, save_path)
            sentence_embeddings = []
            ctr += 1

    if len(sentence_embeddings) > 0:
        sentence_embeddings_tensor = torch.tensor(sentence_embeddings)
        save_path = f"../data/time_frame_emb_{ctr * save_interval}.pt"
        torch.save(sentence_embeddings_tensor, save_path)

    model = None
    gc.collect()

    # model_name = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    #
    # def text2vec(text):
    #     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    #
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #
    #     # 使用 [CLS] token 的输出作为句子的表示
    #     sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()
    #     return sentence_vector
    #
    # # 创建字典以存储句子和其对应的嵌入向量
    # text_sentence_2_embedding = {}
    #
    # for sentence in tqdm(cleaned_sentence_set, desc="Processing sentences", unit="sentence"):
    #     try:
    #         text_sentence_2_embedding[sentence] = text2vec(sentence)
    #     except:
    #         continue
    #
    # output_dir = '../data'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # output_file = os.path.join(output_dir, 'time_frame2embedding.pkl')
    # with open(output_file, 'wb') as f:
    #     pickle.dump(text_sentence_2_embedding, f)
    #
    # print(f"Embedding saved to {output_file}")
    # return

def load_time_frame_2_vec():
    # sentence_2_vec = pickle.load(open('../data/time_frame2embedding.pkl', 'rb'))
    file_paths = ["/time_frame_emb_800000.pt" ]
    # file_paths = ["/sentence_emb.pt"]
    data_path = '../data'
    # 存放加载的张量
    all_embeddings = []

    # 加载每个文件的张量并添加到列表中
    for file_path in file_paths:
        embeddings = torch.load(data_path + file_path)  # 加载 .pt 文件中的张量
        all_embeddings.append(embeddings)  # 将加载的张量添加到列表

    # 将所有张量沿第0维拼接（即按行合并）
    sentence_emb = torch.cat(all_embeddings, dim=0)

    # sentence_emb = torch.load(f"{data_path}/sentence_emb.pt")
    data = json.load(open(f"{data_path}/time_frame2id.json", "r"))

    sentence_2_vec = {sentence: sentence_emb[idx] for sentence, idx in data.items()}
    return sentence_2_vec


def time_frame2embedding(text_lst):
    sentence_2_vec = load_time_frame_2_vec()
    # text_feature = [torch.tensor(sentence_2_vec[sentence]).view(1,-1) for sentence in text_lst if sentence in sentence_2_vec]
    # if text_feature == []:
    #     text_feature = torch.zeros(1,768)
    # else:
    #     text_feature = torch.cat(text_feature, 0)
    # return text_feature# 4, 768/1, 768
    text_emb=[]
    for texts in text_lst:
        texts = eval(texts)
        for i in range(len(texts)):
            texts[i] = clean_text(texts[i])
        if not texts:
            # print("Warning: Empty text is found")
            text_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            try:
                texts_emb, _ = torch.max(
                    torch.stack([sentence_2_vec[text] for text in texts]), dim=0)
                text_emb.append(texts_emb)
                # text_emb.append(sentence_2_vec[text])
            except:
                print("Warning: Error2 text, not emb time_frame")
                text_emb.append(torch.zeros(768, dtype=torch.float32))
    return torch.stack(text_emb)  # len(text_list), 768

if __name__ == "__main__":
    save_sentence_bert_dict_pkl()