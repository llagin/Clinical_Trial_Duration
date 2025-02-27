from preprocess.icdcode_encode import query_embedding_icdcodes
from preprocess.molecule_encode import txt_to_lst
from preprocess.smiless_encode import query_embeddings
from preprocess.text_encode import text2embedding
from preprocess.time_frame_encode import time_frame2embedding
import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from preprocess.drug_disease_encode import feature2embedding
from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec, get_sentence_embedding
import numpy as np
from tqdm import tqdm


train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t',
                         dtype={'masking': str, 'intervention_model': str})
encoder = OneHotEncoder(sparse=False)
encoder.fit(train_data[['phase']])

sentence2vec = load_sentence_2_vec("../data")

def pad_sentences(paragraphs, padding_size):
    padded_paragraphs = []
    mask_matrices = []

    for p in paragraphs:
        num_padding = padding_size - p.size(0)

        if num_padding > 0:
            padding = torch.zeros(num_padding, p.size(1))
            padded_p = torch.cat([p, padding], dim=0)
        else:
            padded_p = p

        # 1 for actual data, 0 for padding
        mask = torch.cat([torch.ones(p.size(0)), torch.zeros(num_padding)], dim=0)

        padded_paragraphs.append(padded_p)
        mask_matrices.append(mask)

    padded_paragraphs_tensor = torch.stack(padded_paragraphs)
    mask_matrices_tensor = torch.stack(mask_matrices)

    return padded_paragraphs_tensor, mask_matrices_tensor


def criteria2embedding(criteria_lst, padding_size):
    criteria_lst = [protocol2feature(criteria, sentence2vec) for criteria in criteria_lst]

    criteria_emb = [torch.cat([torch.mean(criteria[0], 0).view(1, -1), torch.mean(criteria[1], 0).view(1, -1)], dim=1)
                    for criteria in criteria_lst]
    return torch.cat(criteria_emb, dim=0)

    # max_sentences = max(max(p[0].size(0), p[1].size(0)) for p in criteria_lst)
    # if max_sentences < padding_size:
    #     print(
    #         f"Warning: padding size is larger than the maximum number of sentences in the data. Padding size: {padding_size}, Max sentences: {max_sentences}")
    #
    # incl_criteria = [criteria[0][:padding_size] for criteria in criteria_lst]
    # incl_emb, incl_mask = pad_sentences(incl_criteria, padding_size)
    #
    # excl_criteria = [criteria[1][:padding_size] for criteria in criteria_lst]
    # excl_emb, excl_mask = pad_sentences(excl_criteria, padding_size)
    #
    # return incl_emb, incl_mask, excl_emb, excl_mask

def preprocess_num_data(data, column_name):
    mean_value = data[column_name].mean()
    data[column_name].fillna(mean_value if not pd.isna(mean_value) else 0, inplace=True)
    return data

def raw_data_encode(data):
    if isinstance(data, np.ndarray):
        features = ['criteria', 'phase', 'drugs', 'diseases', 'title', 'summary', 'enrollment', 'number_of_location',
                    'intervention_model', 'masking', 'primary_purpose', 'time_frame', 'smiless', 'icdcodes']
        data = pd.DataFrame(data, columns=features)

    # criteria
    padding_size = 32
    data['criteria'].fillna('', inplace=True)
    # incl_emb, incl_mask, excl_emb, excl_mask = criteria2embedding(data['criteria'], padding_size)
    criteria_emb = criteria2embedding(data['criteria'], padding_size)

    # phase
    phase_emb = torch.tensor(encoder.transform(data[['phase']])).float()

    # drugs,diseases
    drug_emb = feature2embedding('drugs', data['drugs'].tolist())
    disease_emb = feature2embedding('diseases', data['diseases'].tolist())

    # title,summary,primary_purpose
    title_emb = text2embedding(data['title'].tolist())
    summary_emb = text2embedding(data['summary'].tolist())
    primary_purpose_emb = text2embedding(data['primary_purpose'].tolist())

    # time_frame
    time_frame_emb = time_frame2embedding(data['time_frame'].tolist())

    # intervention_model,masking
    def convert_to_tensor(binary_string):
        tensor_values = [int(bit) for bit in binary_string]
        return torch.tensor(tensor_values, dtype=torch.float32)

    tensors = data['intervention_model'].fillna('00000').apply(convert_to_tensor)
    intervention_model_emb = torch.stack(tensors.tolist())

    tensors = data['masking'].fillna('0000').apply(convert_to_tensor)
    masking_emb = torch.stack(tensors.tolist())

    # enrollment,number_of_location
    data['enrollment'] = preprocess_num_data(data, 'enrollment')['enrollment']
    enrollment_emb = torch.tensor(data[['enrollment']].values).float()
    data['number_of_location'] = preprocess_num_data(data, 'number_of_location')['number_of_location']
    location_emb = torch.tensor(data[['number_of_location']].values).float()

    # icdcodes
    icd_lst = data['icdcodes'].tolist()
    emb = query_embedding_icdcodes(icd_lst)
    icd_emb = torch.tensor(np.array(emb))

    # smiless
    smiles_lst_lst = data['smiless'].tolist()
    emb = query_embeddings(smiles_lst_lst)
    smiles_emb = torch.tensor(np.array(emb))

    # return incl_emb, incl_mask, excl_emb, excl_mask, phase_emb, drug_emb, disease_emb, title_emb, summary_emb, \
    #     primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, enrollment_emb, location_emb
    return criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb, \
        primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, enrollment_emb, location_emb, \
        smiles_emb, icd_emb

