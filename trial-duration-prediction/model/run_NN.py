# %%
import time
loading_start_time = time.time()
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model import Protocol_Attention_Regression_FACT_new
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from embeddings_cache import EmbeddingCache
from torch.optim.lr_scheduler import CosineAnnealingLR
# from category_encoders import LeaveOneOutEncoder

# %%


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# %%
import os
os.getcwd()
print(f"Loading bag need time:{time.time() - loading_start_time:.4f}s")
cache = EmbeddingCache()
class Trial_Dataset(Dataset):
    def __init__(self, nctid_lst, criteria_emb, drug_emb, dis_emb, phase_emb,
                 title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                 enrollment_emb, location_emb, smiles_emb, icd_emb, target_lst):
        self.nctid_lst = nctid_lst
        self.target_lst = torch.tensor(target_lst.values, dtype=torch.float32)

        self.criteria_emb = criteria_emb
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
        return self.nctid_lst.iloc[idx], self.criteria_emb[idx], self.drug_emb[idx], self.dis_emb[idx], self.phase_emb[idx], \
            self.title_emb[idx], self.summary_emb[idx], self.primary_purpose_emb[idx], self.time_frame_emb[idx], \
            self.intervention_model_emb[idx], self.masking_emb[idx], self.enrollment_emb[idx], self.location_emb[idx], \
            self.smiles_emb[idx], self.icd_emb[idx], self.target_lst[idx]

def test(model, data_loader):
    model.eval()
    good_predictions = 0
    with torch.no_grad():
        predictions = []
        targets = []
        large_errors = []

    for batch_idx, batch_data in enumerate(data_loader):
        # (inclusion_emb, inclusion_mask), (exclusion_emb, exclusion_mask)
        nctid, criteria_emb, drug_emb, disease_emb, phase_emb, \
            title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, \
            enrollment_emb, location_emb, smiles_emb, icd_emb, target = batch_data

        criteria_emb, drug_emb, disease_emb, phase_emb, \
            title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, \
            enrollment_emb, location_emb, smiles_emb, icd_emb, target = \
             criteria_emb.to(device), \
                drug_emb.to(device), disease_emb.to(device), phase_emb.to(device), title_emb.to(device), \
                summary_emb.to(device), primary_purpose_emb.to(device), time_frame_emb.to(device), \
                intervention_model_emb.to(device), masking_emb.to(device), enrollment_emb.to(device), \
                location_emb.to(device), smiles_emb.to(device), icd_emb.to(device), target.to(device)


        output = model.forward(criteria_emb, drug_emb, disease_emb, phase_emb,
                               title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                               enrollment_emb, location_emb, smiles_emb, icd_emb)
        prediction = output[:, 0]

        predictions.extend(prediction.tolist())
        targets.extend(target.tolist())


    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson_score, _ = pearsonr(targets, predictions)
    return mae, mse, r2, pearson_score

use_valid = True

input_data_df = pd.read_csv("../data/new_input.csv", sep=',', dtype={'masking': str, 'intervention_model': str})

input_data_df['start_year'] = input_data_df['start_date'].str[-4:]
input_data_df['completion_year'] = input_data_df['completion_date'].str[-4:]
print(input_data_df.shape)
time_start = '2015'
time_end = '2025'
input_data_df = input_data_df[input_data_df['completion_year'] < time_end]
input_data_df = input_data_df[input_data_df['start_year'] >= time_start]

time_ = '2021'
train_data = input_data_df[input_data_df['completion_year'] < time_]
test_data = input_data_df[input_data_df['start_year'] >= time_]

print(train_data.shape)
print(test_data.shape)
train_data = train_data[train_data['criteria'].isnull() == False]
test_data = test_data[test_data['criteria'].isnull() == False]
if use_valid:
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
print(train_data.head())
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
get_embedding_start_time = time.time()
criteria_emb['train'], phase_emb['train'], drug_emb['train'], disease_emb['train'], title_emb['train'], summary_emb['train'], primary_purpose_emb['train'], time_frame_emb['train'], intervention_model_emb['train'], masking_emb['train'], enrollment_emb['train'], location_emb['train'], smiles_emb['train'], icd_emb['train'] = cache.query_df(train_data)
print('get train_emb')
criteria_emb['test'], phase_emb['test'], drug_emb['test'], disease_emb['test'], title_emb['test'], summary_emb['test'], primary_purpose_emb['test'], time_frame_emb['test'], intervention_model_emb['test'], masking_emb['test'], enrollment_emb['test'], location_emb['test'], smiles_emb['test'], icd_emb['test'] = cache.query_df(test_data)
print('get test_emb')
if use_valid:
    criteria_emb['valid'], phase_emb['valid'], drug_emb['valid'], disease_emb['valid'], title_emb['valid'], \
        summary_emb['valid'], primary_purpose_emb['valid'], time_frame_emb['valid'], \
        intervention_model_emb['valid'], masking_emb['valid'], enrollment_emb['valid'], location_emb['valid'], \
        smiles_emb['valid'], icd_emb['valid'] = cache.query_df(valid_data)
    print('get valid_emb')
print(f'Loading embeddings time:{time.time() - get_embedding_start_time}s')
# %%
batch_size = 256
train_dataset = Trial_Dataset(train_data['nctid'], criteria_emb['train'], drug_emb['train'], disease_emb['train'], phase_emb['train'],
                              title_emb['train'], summary_emb['train'], primary_purpose_emb['train'],
                              time_frame_emb['train'], intervention_model_emb['train'], masking_emb['train'],
                              enrollment_emb['train'], location_emb['train'], smiles_emb['train'], icd_emb['train'], train_data['time_day'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Trial_Dataset(test_data['nctid'], criteria_emb['test'], drug_emb['test'], disease_emb['test'], phase_emb['test'],
                             title_emb['test'], summary_emb['test'], primary_purpose_emb['test'],
                             time_frame_emb['test'], intervention_model_emb['test'], masking_emb['test'],
                             enrollment_emb['test'], location_emb['test'], smiles_emb['test'], icd_emb['test'], test_data['time_day'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# %%
if use_valid:
    valid_dataset = Trial_Dataset(valid_data['nctid'], criteria_emb['valid'], drug_emb['valid'], disease_emb['valid'], phase_emb['valid'],
                                  title_emb['valid'], summary_emb['valid'], primary_purpose_emb['valid'],
                                  time_frame_emb['valid'], intervention_model_emb['valid'], masking_emb['valid'],
                                  enrollment_emb['valid'], location_emb['valid'], smiles_emb['valid'], icd_emb['valid'], valid_data['time_day'])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# %%
mae_list = []
mse_list = []
r2_list = []
pearson_list = []
print("Using device:", device)
print(device)
for i in range(5):
    num_epochs = 1000
    random_offset = random.randint(0, 10000)
    torch.manual_seed(random_offset)
    protocol_model = Protocol_Attention_Regression_FACT_new(1)
    protocol_model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.NAdam(
        protocol_model.parameters(),
        lr=5e-4,
        weight_decay=0.001
    )

    with SummaryWriter(f'logs/NN_model_log') as writer:
        print("Start training")
        best_mse = float('inf')
        epochs_no_improve = 0
        patience = 20
        best_score = float('inf')
        for epoch in tqdm(range(num_epochs)):
            protocol_model.train()
            for batch_idx, batch_data in enumerate(train_loader):
                nctid, criteria_emb, drug_emb, disease_emb, phase_emb, \
                    title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, \
                    enrollment_emb, location_emb, smiles_emb, icd_emb, target = batch_data
                criteria_emb, drug_emb, disease_emb, phase_emb, \
                    title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, \
                    enrollment_emb, location_emb, smiles_emb, icd_emb, target = \
                    criteria_emb.to(device), \
                        drug_emb.to(device), disease_emb.to(device), phase_emb.to(device), title_emb.to(device), \
                        summary_emb.to(device), primary_purpose_emb.to(device), time_frame_emb.to(device), \
                        intervention_model_emb.to(device), masking_emb.to(device), enrollment_emb.to(device), \
                        location_emb.to(device), smiles_emb.to(device), icd_emb.to(device), target.to(device)
                output = protocol_model.forward(criteria_emb, drug_emb, disease_emb, phase_emb,
                                       title_emb, summary_emb, primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                                       enrollment_emb, location_emb, smiles_emb, icd_emb)
                prediction = output[:, 0]
                loss = criterion(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + batch_idx)
            if use_valid:
                valid_mae, valid_mse, _, _ = test(protocol_model, valid_loader)
                writer.add_scalar('valid_MAE', valid_mae, epoch)
                writer.add_scalar('valid_MSE', valid_mse, epoch)
            else:
                test_mae, test_mse, _, _, = test(protocol_model, test_loader)
                writer.add_scalar('MAE', test_mae, epoch)
                writer.add_scalar('MSE', test_mse, epoch)
            train_mae, train_mse, _, _ = test(protocol_model, train_loader)
            writer.add_scalar('train_MAE', train_mae, epoch)
            writer.add_scalar('train_MSE', train_mse, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            score = valid_mae
            if use_valid and score < best_score:
                best_score = score
                epochs_no_improve = 0
                torch.save(protocol_model.state_dict(), f'checkpoints/mlp_checkpoint_15_21_{i}.pt')
            elif use_valid:
                epochs_no_improve += 1

            if use_valid and epochs_no_improve >= patience:
                break
    
    protocol_model.load_state_dict(torch.load(f'checkpoints/mlp_checkpoint_15_21_{i}.pt'))
    mae, mse, r2, pearson_score = test(protocol_model, test_loader)
    print(f'Test MAE: {mae:.3f}')
    print(f'Test MSE: {mse:.3f}')
    print(f'Test r2 score: {r2:.3f}')
    print(f'Test pearson score: {pearson_score:.3f}')
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    pearson_list.append(pearson_score)
mae_arr = np.array(mae_list)
mse_arr = np.array(mse_list)
rmse_arr = np.sqrt(mse_arr)
r2_arr = np.array(r2_list)
pearson_list = np.array(pearson_list)
print(f"MAE: {mae_arr.mean():.3f} ({mae_arr.std():.3f})")
print(f"MSE: {mse_arr.mean():.3f} ({mse_arr.std():.3f})")
print(f"RMSE: {rmse_arr.mean():.3f} ({rmse_arr.std():.3f})")
print(f"R2: {r2_arr.mean():.3f} ({r2_arr.std():.3f})")
print(f"Pearson: {pearson_list.mean():.3f} ({pearson_list.std():.3f})")