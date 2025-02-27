# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from models import Protocol_Embedding_Regression, Protocol_Attention_Regression
from torch_lr_finder import LRFinder

# %%
import sys

sys.path.append('../')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import os

os.getcwd()


class Trial_Dataset(Dataset):
    def __init__(self, nctid_lst, criteria_emb, drug_emb, dis_emb, phase_emb,
                 title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb,
                 masking_emb, enrollment_emb, location_emb, smiles_emb, icd_emb, target_lst):
        self.nctid_lst = nctid_lst
        self.target_lst = torch.tensor(target_lst.values, dtype=torch.float32)

        self.criteria_emb = criteria_emb
        # self.incl_mask = incl_mask
        #
        # self.excl_emb = excl_emb
        # self.excl_mask = excl_mask
        self.drug_emb = drug_emb
        self.dis_emb = dis_emb
        self.phase_emb = phase_emb
        self.title_emb = title_emb
        self.summary_emb = summary_emb
        self.primary_purpose_emb = primary_purpose_emb
        self.time_frame_emb = time_frame_emb
        self.intervention_model_emb = intervention_model_emb
        self.masking_emb = masking_emb
        self.enrollment_emb = enrollment_emb
        self.location_emb = location_emb
        self.smiles_emb = smiles_emb
        self.icd_emb = icd_emb

    def __len__(self):
        return len(self.nctid_lst)

    def __getitem__(self, idx):
        return (self.nctid_lst.iloc[idx], self.criteria_emb[idx], self.drug_emb[idx], self.dis_emb[idx],
                self.phase_emb[idx], self.title_emb[idx], self.summary_emb[idx], self.primary_purpose_emb[idx],
                self.time_frame_emb[idx], self.intervention_model_emb[idx], self.masking_emb[idx],
                self.enrollment_emb[idx], self.location_emb[idx], self.smiles_emb[idx], self.icd_emb[idx]), self.target_lst[idx]


if __name__ == '__main__':
    use_valid = True

    # %%
    train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t',
                             dtype={'masking': str, 'intervention_model': str})

    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
    print(train_data.head())

    from preprocess.raw_data_encode import raw_data_encode
    # incl_emb = {}
    # incl_mask = {}
    # excl_emb = {}
    # excl_mask = {}
    criteria_emb = {}
    phase_emb = {}
    drug_emb = {}
    disease_emb = {}
    title_emb = {}
    summary_emb = {}
    primary_purpose_emb = {}
    time_frame_emb = {}
    intervention_model_emb = {}
    masking_emb = {}
    enrollment_emb = {}
    location_emb = {}
    smiles_emb = {}
    icd_emb = {}
    criteria_emb['train'], drug_emb['train'], disease_emb['train'], phase_emb['train'], title_emb['train'], \
        summary_emb['train'], primary_purpose_emb['train'], \
        time_frame_emb['train'], intervention_model_emb['train'], masking_emb['train'], enrollment_emb['train'], \
        location_emb['train'], smiles_emb['train'], icd_emb['train'] = raw_data_encode(train_data)
    print('get train_emb')
    # %%
    batch_size = 256

    train_dataset = Trial_Dataset(train_data['nctid'], criteria_emb['train'], drug_emb['train'], disease_emb['train'],
                                  phase_emb['train'],
                                  title_emb['train'], summary_emb['train'], primary_purpose_emb['train'],
                                  time_frame_emb['train'], intervention_model_emb['train'], masking_emb['train'],
                                  enrollment_emb['train'], location_emb['train'], smiles_emb['train'], icd_emb['train'],
                                  train_data['time_day'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    criteria_emb['valid'], drug_emb['valid'], disease_emb['valid'], phase_emb['valid'], title_emb['valid'], \
        summary_emb['valid'], primary_purpose_emb['valid'], time_frame_emb['valid'], \
        intervention_model_emb['valid'], masking_emb['valid'], enrollment_emb['valid'], location_emb['valid'], \
        smiles_emb['valid'], icd_emb['valid'] = raw_data_encode(valid_data)
    print('get valid_emb')

    valid_dataset = Trial_Dataset(valid_data['nctid'], criteria_emb['valid'], drug_emb['valid'], disease_emb['valid'], phase_emb['valid'],
                                  title_emb['valid'], summary_emb['valid'], primary_purpose_emb['valid'],
                                  time_frame_emb['valid'], intervention_model_emb['valid'], masking_emb['valid'],
                                  enrollment_emb['valid'], location_emb['valid'], smiles_emb['valid'], icd_emb['valid'], valid_data['time_day'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    num_epochs = 100

    # protocol_model = Protocol_Embedding_Regression(output_dim=1)
    torch.manual_seed(0)
    protocol_model = Protocol_Attention_Regression(sentence_embedding_dim=768, linear_output_dim=1,
                                                   transformer_encoder_layers=2, num_heads=8, dropout=0.1,
                                                   pooling_method="cls")
    protocol_model.to(device)

    optimizer = optim.AdamW(protocol_model.parameters(), lr=1e-6, weight_decay=0.001)
    criterion = nn.MSELoss()

    lr_finder = LRFinder(protocol_model, optimizer, criterion, device=device)

    print("Finding optimal learning rate...")
    lr_finder.range_test(train_loader, valid_loader, end_lr=0.5, num_iter=300)  # 可以调整 num_iter 为你的训练迭代次数

    lr_finder.plot()

    # 获取最佳学习率
    lr_finder.reset()

