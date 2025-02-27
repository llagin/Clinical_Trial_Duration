# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from models import Protocol_Embedding_Regression, Protocol_Attention_Regression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from transformers import AutoTokenizer, AutoModel
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# from category_encoders import LeaveOneOutEncoder


# %%
import sys
sys.path.append('../')

from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec, get_sentence_embedding
from preprocess.drug_disease_encode import feature2embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
import os
os.getcwd()
# %%
# sentence2vec = load_sentence_2_vec("../data")

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
        return self.nctid_lst.iloc[idx], self.criteria_emb[idx], self.drug_emb[idx], self.dis_emb[idx], self.phase_emb[idx], \
            self.title_emb[idx], self.summary_emb[idx], self.primary_purpose_emb[idx], self.time_frame_emb[idx], \
            self.intervention_model_emb[idx], self.masking_emb[idx], self.enrollment_emb[idx], self.location_emb[idx], \
            self.smiles_emb[idx], self.icd_emb[idx], self.target_lst[idx]

    # (self.incl_emb[idx], self.incl_mask[idx]), (
    #     self.excl_emb[idx], self.excl_mask[idx])


# %%
# def drug2embedding(drug_lst):
#     model_name = "dmis-lab/biobert-base-cased-v1.2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#     drug_emb = []
#     for drugs in tqdm(drug_lst):
#         if len(drugs) == 0:
#             print("Warning: Empty drug list is found")
#             drug_emb.append(torch.zeros(768, dtype=torch.float32))
#         else:
#             # mean pooling
#             drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug, tokenizer, model) for drug in drugs.split(';')]), dim=0)
#             drug_emb.append(drugs_emb)
#
#     return torch.stack(drug_emb)
#
# def disease2embedding(disease_lst):
#     model_name = "dmis-lab/biobert-base-cased-v1.2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     disease_emb = []
#     for diseases in tqdm(disease_lst):
#         if len(diseases) == 0:
#             print("Warning: Empty disease list is found")
#             disease_emb.append(torch.zeros(768, dtype=torch.float32))
#         else:
#             # mean pooling
#             diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease, tokenizer, model) for disease in diseases.split(';')]), dim=0)
#             disease_emb.append(diseases_emb)
#
#     return torch.stack(disease_emb) # len(disease_list), 768
# %%

def test(model, data_loader):
    model.eval()

    with torch.no_grad():
        predictions = []
        targets = []

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


        output = model.forward(criteria_emb, drug_emb, disease_emb,
                               phase_emb, title_emb, summary_emb, primary_purpose_emb, time_frame_emb,
                               intervention_model_emb,
                               masking_emb, enrollment_emb, location_emb, smiles_emb, icd_emb)
        prediction = output[:, 0]

        predictions.extend(prediction.tolist())
        targets.extend(target.tolist())

    # print('predictions', np.isnan(predictions).sum())  # 检查 predictions 中 NaN 的数量

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson_score, _ = pearsonr(targets, predictions)

    return mae, mse, r2, pearson_score

if __name__ == '__main__':
    use_valid = True

    # %%
    train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t',
                             dtype={'masking': str, 'intervention_model': str})
    test_data = pd.read_csv(f'../data/time_prediction_test.csv', sep='\t',
                            dtype={'masking': str, 'intervention_model': str})

    if use_valid:
        train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
    print(train_data.head())

    # %%
    # if not os.path.exists('../data/drug_emb.pt') or not os.path.exists('../data/disease_emb.pt'):
    #     drug_emb = {}
    #     drug_emb['train'] = drug2embedding(train_data['drugs'].tolist())
    #     drug_emb['test'] = drug2embedding(test_data['drugs'].tolist())
    #
    #     disease_emb = {}
    #     disease_emb['train'] = disease2embedding(train_data['diseases'].tolist())
    #     disease_emb['test'] = disease2embedding(test_data['diseases'].tolist())
    #
    #     if use_valid:
    #         drug_emb['valid'] = drug2embedding(valid_data['drugs'].tolist())
    #         disease_emb['valid'] = disease2embedding(valid_data['diseases'].tolist())
    #
    #     torch.save(drug_emb, '../data/drug_emb.pt')
    #     torch.save(disease_emb, '../data/disease_emb.pt')
    # else:
    #     drug_emb = torch.load('../data/drug_emb.pt')
    #     disease_emb = torch.load('../data/disease_emb.pt')

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
        summary_emb['train'], primary_purpose_emb['train'],\
        time_frame_emb['train'], intervention_model_emb['train'], masking_emb['train'], enrollment_emb['train'], \
        location_emb['train'], smiles_emb['train'], icd_emb['train'] = raw_data_encode(train_data)
    print('get train_emb')
    criteria_emb['test'], drug_emb['test'], disease_emb['test'], phase_emb['test'], title_emb['test'], \
        summary_emb['test'], primary_purpose_emb['test'],\
        time_frame_emb['test'], intervention_model_emb['test'], masking_emb['test'], enrollment_emb['test'], \
        location_emb['test'], smiles_emb['test'], icd_emb['test'] = raw_data_encode(test_data)
    print('get test_emb')
    if use_valid:
        criteria_emb['valid'], drug_emb['valid'], disease_emb['valid'], phase_emb['valid'], title_emb['valid'], \
            summary_emb['valid'], primary_purpose_emb['valid'], time_frame_emb['valid'], \
            intervention_model_emb['valid'], masking_emb['valid'], enrollment_emb['valid'], location_emb['valid'], \
            smiles_emb['valid'], icd_emb['valid'] = raw_data_encode(valid_data)
        print('get valid_emb')

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
        # valid_data['criteria'].fillna('', inplace=True)
        # incl_emb['valid'], incl_mask['valid'], excl_emb['valid'], excl_mask['valid'] = criteria2embedding(
        #     valid_data['criteria'], padding_size)
        # phase_emb['valid'] = torch.tensor(encoder.transform(valid_data[['phase']])).float()
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
    for i in range(5):
        num_epochs = 300

        # protocol_model = Protocol_Embedding_Regression(output_dim=1)
        torch.manual_seed(i)
        protocol_model = Protocol_Attention_Regression(sentence_embedding_dim=768, linear_output_dim=1,
                                                       transformer_encoder_layers=2, num_heads=8, dropout=0.1,
                                                       pooling_method="cls")
        # print(protocol_model)

        protocol_model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(protocol_model.parameters(), lr=8.33E-04, weight_decay=0.001)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

        # Create a SummaryWriter instance
        with SummaryWriter(f'logs/NN_model_log') as writer:
            print("Start training")
            best_mse = float('inf')
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

                    output = protocol_model.forward(criteria_emb, drug_emb, disease_emb,
                                           phase_emb, title_emb, summary_emb, primary_purpose_emb, time_frame_emb,
                                           intervention_model_emb,
                                           masking_emb, enrollment_emb, location_emb, smiles_emb, icd_emb)
                    prediction = output[:, 0]
                    loss = criterion(prediction, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Write the loss to TensorBoard
                    if batch_idx % 50 == 0:
                        writer.add_scalar('Loss', loss.item(), epoch * len(train_loader) + batch_idx)

                if epoch % 5 == 0:
                    if use_valid:
                        valid_mae, valid_mse, _, _ = test(protocol_model, valid_loader)
                        writer.add_scalar('valid_MAE', valid_mae, epoch)
                        writer.add_scalar('valid_MSE', valid_mse, epoch)
                    else:
                        test_mae, test_mse, _, _ = test(protocol_model, test_loader)
                        writer.add_scalar('MAE', test_mae, epoch)
                        writer.add_scalar('MSE', test_mse, epoch)

                    train_mae, train_mse, _, _ = test(protocol_model, train_loader)
                    writer.add_scalar('train_MAE', train_mae, epoch)
                    writer.add_scalar('train_MSE', train_mse, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                    # if epoch < 10:
                    #     scheduler.step()

                    if valid_mse < best_mse:
                        best_mse = valid_mse
                        torch.save(protocol_model.state_dict(), f'checkpoints/mlp_checkpoint{i}.pt')

        protocol_model.load_state_dict(torch.load(f'checkpoints/mlp_checkpoint{i}.pt'))
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
