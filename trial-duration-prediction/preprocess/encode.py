import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('../')
import h5py
from preprocess.icdcode_encode import query_embedding_icdcodes
from preprocess.molecule_encode import txt_to_lst
from preprocess.smiless_encode import query_embeddings
from preprocess.text_encode import text2embedding, load_sentence_2_vec as load_text_2_vec, clean_text
from preprocess.time_frame_encode import time_frame2embedding, load_time_frame_2_vec, clean_text as clean_time_frame
from sklearn.preprocessing import OneHotEncoder
import torch
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from preprocess.drug_disease_encode import feature2embedding, load_feature_2_vec, clean_text as clean_feature
from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec as load_criteria_2_vec, get_sentence_embedding, clean_protocol, split_protocol
import numpy as np
from tqdm import tqdm
from multiprocessing import Manager
import os
import warnings
warnings.filterwarnings("ignore")

# sentence2vec = load_sentence_2_vec("../data")
device = torch.device("cpu")
train_data = pd.read_csv(f'../data/new_input.csv', sep=',',
                         dtype={'masking': str, 'intervention_model': str})
train_data['primary_purpose'].fillna('missing',inplace=True)
feature = ['criteria', 'phase', 'drugs', 'diseases', 'title', 'summary', 'enrollment', 'number_of_location', 'intervention_model', 'masking', 'primary_purpose', 'time_frame', 'smiless', 'icdcodes']
train_data = train_data[feature]

class CriteriaEncoder:
    def __init__(self, data_path="../data", device="cpu"):
        self.device = device
        self.sentence_2_vec = load_criteria_2_vec(data_path)
        for k in self.sentence_2_vec:
            self.sentence_2_vec[k] = self.sentence_2_vec[k].to(self.device)

    def encode(self, criteria_lst, padding_size=32):
        emb_list = []
        for criteria in criteria_lst:
            inclusion_criteria, exclusion_criteria = split_protocol(criteria)
            inclusion_feature = [self.sentence_2_vec[s] for s in inclusion_criteria if s in self.sentence_2_vec]
            exclusion_feature = [self.sentence_2_vec[s] for s in exclusion_criteria if s in self.sentence_2_vec]

            inclusion_emb = torch.mean(torch.stack(inclusion_feature), 0) if inclusion_feature else torch.zeros(768, device=self.device)
            exclusion_emb = torch.mean(torch.stack(exclusion_feature), 0) if exclusion_feature else torch.zeros(768, device=self.device)

            emb_list.append(torch.cat([inclusion_emb, exclusion_emb], dim=0).view(1, -1))
        return torch.cat(emb_list, dim=0)
    
    def encode_single(self, criteria):
        c = protocol2feature(criteria, self.sentence_2_vec)
        criteria_emb = torch.cat([torch.mean(c[0], 0).view(1, -1), torch.mean(c[1], 0).view(1, -1)], dim=1)
        return criteria_emb.squeeze(0)

class FeatureEncoder:
    def __init__(self, feature, data_path="../data", device="cpu"):
        self.feature = feature
        self.device = device
        self.sentence_2_vec = load_feature_2_vec(feature)
        for k in self.sentence_2_vec:
            self.sentence_2_vec[k] = self.sentence_2_vec[k].to(self.device)

    def encode(self, text_lst):
        emb_list = []
        for text in text_lst:
            cleaned_text = clean_feature(text)
            if cleaned_text in self.sentence_2_vec:
                emb_list.append(self.sentence_2_vec[cleaned_text])
            else:
                emb_list.append(torch.zeros(768, device=self.device))
        return torch.stack(emb_list)
    
    def encode_single(self, text):
        text = clean_feature(text)
        if not text:
            return torch.zeros(768, dtype=torch.float32)
        else:
            try:
                return self.sentence_2_vec[text]
            except:
                return torch.zeros(768, dtype=torch.float32)
                
class TextEncoder:
    def __init__(self, data_path="../data", device="cpu"):
        self.device = device
        self.sentence_2_vec = load_text_2_vec()
        for k in self.sentence_2_vec:
            self.sentence_2_vec[k] = self.sentence_2_vec[k].to(self.device)

    def encode(self, text_lst):
        emb_list = []
        for text in text_lst:
            text = clean_text(text)
            if text in self.sentence_2_vec:
                emb_list.append(self.sentence_2_vec[text])
            else:
                emb_list.append(torch.zeros(768, device=self.device))
        return torch.stack(emb_list)
    
    def encode_single(self, text):
        text = clean_text(text)
        if not text:
            return torch.zeros(768, dtype=torch.float32)
        else:
            try:
                return self.sentence_2_vec[text]
            except:
                return torch.zeros(768, dtype=torch.float32)

class TimeFrameEncoder:
    def __init__(self, data_path="../data", device="cpu"):
        self.device = device
        self.sentence_2_vec = load_time_frame_2_vec()
        for k in self.sentence_2_vec:
            self.sentence_2_vec[k] = self.sentence_2_vec[k].to(self.device)

    def encode(self, text_lst):
        emb_list = []
        for texts in text_lst:
            try:
                texts = eval(texts)
                texts = [clean_time_frame(t) for t in texts]
            except:
                texts = []
            if texts:
                valid_emb = [self.sentence_2_vec[t] for t in texts if t in self.sentence_2_vec]
                if valid_emb:
                    texts_emb, _ = torch.max(torch.stack(valid_emb), dim=0)
                    emb_list.append(texts_emb)
                else:
                    emb_list.append(torch.zeros(768, device=self.device))
            else:
                emb_list.append(torch.zeros(768, device=self.device))
        return torch.stack(emb_list)

    def encode_single(self, texts):
        texts = eval(texts)
        for i in range(len(texts)):
            texts[i] = clean_time_frame(texts[i])
        if not texts:
            return torch.zeros(768, dtype=torch.float32)
        else:
            try:
                texts_emb, _ = torch.max(torch.stack([self.sentence_2_vec[t] for t in texts]), dim=0)
                return texts_emb
            except:
                return torch.zeros(768, dtype=torch.float32)

_global_criteria_encoder = None
_global_drugs_encoder = None
_global_diseases_encoder = None
_global_text_encoder = None
_global_time_frame_encoder = None

def _initialize_global_encoders():
    global _global_criteria_encoder, _global_drugs_encoder, _global_diseases_encoder, \
           _global_text_encoder, _global_time_frame_encoder
    
    if _global_criteria_encoder is None:
        _global_criteria_encoder = CriteriaEncoder(data_path="../data", device="cpu")
        _global_drugs_encoder = FeatureEncoder(feature='drugs', data_path="../data", device="cpu")
        _global_diseases_encoder = FeatureEncoder(feature='diseases', data_path="../data", device="cpu")
        _global_text_encoder = TextEncoder(data_path="../data", device="cpu")
        _global_time_frame_encoder = TimeFrameEncoder(data_path="../data", device="cpu")

# save to HDF5
def save_embeddings_to_hdf5(embeddings_dict, file_path):
    expected_keys = {
        'criteria_emb', 'phase_emb', 'drug_emb', 'disease_emb', 'title_emb',
        'summary_emb', 'primary_purpose_emb', 'time_frame_emb', 'intervention_model_emb',
        'masking_emb', 'enrollment_emb', 'location_emb', 'smiles_emb', 'icd_emb'
    }
    with h5py.File(file_path, 'a') as f:
        for nctid, embedding_dict in embeddings_dict.items():
            missing_keys = expected_keys - set(embedding_dict.keys())
            if missing_keys:
                raise ValueError(f"[{nctid}] missing key: {missing_keys}")

            if nctid in f:
                del f[nctid]

            grp = f.create_group(nctid)
            for name in expected_keys:
                emb = embedding_dict[name]
                grp.create_dataset(name, data=emb.cpu().numpy())

            written_keys = set(grp.keys())
            if written_keys != expected_keys:
                raise RuntimeError(f"[{nctid}] error")

# loading embeddings from HDF5
def load_embeddings_from_hdf5(df, file_path):
    nctid_list = df['nctid'].tolist()
    criteria_emb, phase_emb, drug_emb, disease_emb, title_emb, summary_emb, \
    primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, \
    enrollment_emb, location_emb, smiles_emb, icd_emb = ([] for _ in range(14))

    with h5py.File(file_path, 'r') as f:
        for nctid in nctid_list:
            if nctid in f:
                group = f[nctid]
                criteria_emb.append(torch.tensor(group['criteria_emb'][()]))
                phase_emb.append(torch.tensor(group['phase_emb'][()]))
                drug_emb.append(torch.tensor(group['drug_emb'][()]))
                disease_emb.append(torch.tensor(group['disease_emb'][()]))
                title_emb.append(torch.tensor(group['title_emb'][()]))
                summary_emb.append(torch.tensor(group['summary_emb'][()]))
                primary_purpose_emb.append(torch.tensor(group['primary_purpose_emb'][()]))
                time_frame_emb.append(torch.tensor(group['time_frame_emb'][()]))
                intervention_model_emb.append(torch.tensor(group['intervention_model_emb'][()]))
                masking_emb.append(torch.tensor(group['masking_emb'][()]))
                enrollment_emb.append(torch.tensor(group['enrollment_emb'][()]))
                location_emb.append(torch.tensor(group['location_emb'][()]))
                smiles_emb.append(torch.tensor(group['smiles_emb'][()]))
                icd_emb.append(torch.tensor(group['icd_emb'][()]))
            else:
                raise KeyError(f"{nctid} not found in HDF5 cache.")
    return (criteria_emb, phase_emb, drug_emb, disease_emb, title_emb, summary_emb,
            primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
            enrollment_emb, location_emb, smiles_emb, icd_emb)

def query_single_embedding(nctid, file_path):
    with h5py.File(file_path, 'r') as f:
        if nctid in f:
            group = f[nctid]
            embedding = {name: torch.tensor(group[name][()]) for name in group.keys()}
            return embedding
        else:
            raise KeyError(f"{nctid} not found in HDF5 cache.")

def phase_to_multihot(phase_str):
    mapping = {
        'phase 1': [1, 0, 0, 0],
        'phase 2': [0, 1, 0, 0],
        'phase 3': [0, 0, 1, 0],
        'phase 4': [0, 0, 0, 1],
        'phase 1/phase 2': [1, 1, 0, 0],
        'phase 2/phase 3': [0, 1, 1, 0],
        }
    return mapping.get(phase_str.lower(), [0, 0, 0, 0])

def process_row(row, encoder):
    nctid = row['nctid']
    try:
        criteria_emb = _global_criteria_encoder.encode_single(row['criteria'])
        phase_emb = torch.tensor(phase_to_multihot(row['phase'])).float().to(device)
        drug_emb = _global_drugs_encoder.encode_single(row['drugs'])
        disease_emb = _global_diseases_encoder.encode_single(row['diseases'])
        title_emb = _global_text_encoder.encode_single(row['title'])
        summary_emb = _global_text_encoder.encode_single(row['summary'])
        primary_purpose_emb = torch.tensor(encoder.transform([[row['primary_purpose']]])).float()[0].to(device)
        time_frame_emb = _global_time_frame_encoder.encode_single(row['time_frame'])
        intervention_model_emb = torch.tensor([int(b) for b in row['intervention_model'] or '00000']).float().to(device)
        masking_emb = torch.tensor([int(b) for b in row['masking'] or '0000']).float().to(device)
        enrollment_emb = torch.tensor([row['enrollment'] or 0.0]).float().to(device)
        location_emb = torch.tensor([row['number_of_location'] or 0.0]).float().to(device)
        smiles_emb = torch.tensor(query_embeddings(txt_to_lst(row['smiless']))).float().to(device)
        icd_emb = torch.tensor(query_embedding_icdcodes(row['icdcodes'])).float().to(device)
        emb_dict = {
            'criteria_emb': criteria_emb,
            'phase_emb': phase_emb,
            'drug_emb': drug_emb,
            'disease_emb': disease_emb,
            'title_emb': title_emb,
            'summary_emb': summary_emb,
            'primary_purpose_emb': primary_purpose_emb,
            'time_frame_emb': time_frame_emb,
            'intervention_model_emb': intervention_model_emb,
            'masking_emb': masking_emb,
            'enrollment_emb': enrollment_emb,
            'location_emb': location_emb,
            'smiles_emb': smiles_emb,
            'icd_emb': icd_emb
        }
        return nctid, emb_dict
    except Exception as e:
        print(f"{nctid} error: {e}")
        return nctid, None

def raw_data_encode(data, cache_path='embeddings_cache.h5'):
    assert 'nctid' in data.columns, "the data should have nctid"
    _initialize_global_encoders()
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(train_data[['primary_purpose']])
    if os.path.exists(cache_path):
        with h5py.File(cache_path, 'r') as f:
            cached_nctids = set(f.keys())
    else:
        cached_nctids = set()

    new_data = data[~data['nctid'].isin(cached_nctids)]

    if not new_data.empty:
        new_embeddings = {}
        
        for i, (_, row) in enumerate(new_data.iterrows(), 1):
            nctid, emb_dict = process_row(row, encoder=encoder)
            if emb_dict is not None:
                new_embeddings[nctid] = emb_dict
            
            if i % 100 == 0:
                print(f"Processed {i} rows")
                save_embeddings_to_hdf5(new_embeddings, cache_path)
                new_embeddings = {}
                sys.stdout.flush()

        if new_embeddings:
            save_embeddings_to_hdf5(new_embeddings, cache_path)

def query_embedding_(nctid, file_path):
    with h5py.File(file_path, 'r') as f:
        if nctid not in f:
            raise KeyError(f"{nctid} not found in HDF5 cache.")
        group = f[nctid]
        emb_dict = {name: torch.tensor(group[name][()]) for name in group.keys()}
    return emb_dict

class EmbeddingCache:
    def __init__(self, file_path='../data/embeddings_cache.h5'):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')

    def query(self, nctid):
        if nctid not in self.file:
            raise KeyError(f"{nctid} not found in HDF5 cache.")
        group = self.file[nctid]

        return (
            torch.tensor(group['criteria_emb'][()]),
            torch.tensor(group['phase_emb'][()]),
            torch.tensor(group['drug_emb'][()]),
            torch.tensor(group['disease_emb'][()]),
            torch.tensor(group['title_emb'][()]),
            torch.tensor(group['summary_emb'][()]),
            torch.tensor(group['primary_purpose_emb'][()]),
            torch.tensor(group['time_frame_emb'][()]),
            torch.tensor(group['intervention_model_emb'][()]),
            torch.tensor(group['masking_emb'][()]),
            torch.tensor(group['enrollment_emb'][()]),
            torch.tensor(group['location_emb'][()]),
            torch.tensor(group['smiles_emb'][()]),
            torch.tensor(group['icd_emb'][()])
        )
    
    def query_df(self, df):
        nctid_list = df['nctid'].tolist()
        criteria_emb_list, phase_emb_list, drug_emb_list, disease_emb_list = [], [], [], []
        title_emb_list, summary_emb_list, primary_purpose_emb_list, time_frame_emb_list = [], [], [], []
        intervention_model_emb_list, masking_emb_list, enrollment_emb_list = [], [], []
        location_emb_list, smiles_emb_list, icd_emb_list = [], [], []

        for nctid in nctid_list:
            try:
                (criteria_emb, phase_emb, drug_emb, disease_emb, title_emb, summary_emb,
                primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                enrollment_emb, location_emb, smiles_emb, icd_emb) = self.query(nctid)

                criteria_emb_list.append(criteria_emb)
                phase_emb_list.append(phase_emb)
                drug_emb_list.append(drug_emb)
                disease_emb_list.append(disease_emb)
                title_emb_list.append(title_emb)
                summary_emb_list.append(summary_emb)
                primary_purpose_emb_list.append(primary_purpose_emb)
                time_frame_emb_list.append(time_frame_emb)
                intervention_model_emb_list.append(intervention_model_emb)
                masking_emb_list.append(masking_emb)
                enrollment_emb_list.append(enrollment_emb)
                location_emb_list.append(location_emb)
                smiles_emb_list.append(smiles_emb)
                icd_emb_list.append(icd_emb)
            except KeyError as e:
                print(f"{nctid} not found in cache, skipped.")

        # 拼接每类向量，dim=0
        return (
            torch.stack(criteria_emb_list, dim=0),
            torch.stack(phase_emb_list, dim=0),
            torch.stack(drug_emb_list, dim=0),
            torch.stack(disease_emb_list, dim=0),
            torch.stack(title_emb_list, dim=0),
            torch.stack(summary_emb_list, dim=0),
            torch.stack(primary_purpose_emb_list, dim=0),
            torch.stack(time_frame_emb_list, dim=0),
            torch.stack(intervention_model_emb_list, dim=0),
            torch.stack(masking_emb_list, dim=0),
            torch.stack(enrollment_emb_list, dim=0),
            torch.stack(location_emb_list, dim=0),
            torch.stack(smiles_emb_list, dim=0),
            torch.stack(icd_emb_list, dim=0)
        )


if __name__ == "__main__":
    data = pd.read_csv(f'../data/new_input.csv', sep=',',
                         dtype={'masking': str, 'intervention_model': str})
    data['primary_purpose'].fillna('missing', inplace=True)
    data['enrollment'].fillna(0.0, inplace=True)
    data['number_of_location'].fillna(0.0, inplace=True)
    data['criteria'] = data['criteria'].fillna('').astype(str)
    nctid_list = data['nctid'].tolist()
    # cache = EmbeddingCache(file_path='../data/embeddings_cache.h5')
    cache_path = '../data/embeddings_cache.h5'
    
    raw_data_encode(data, cache_path=cache_path)
