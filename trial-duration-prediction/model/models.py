from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
import numpy as np
import pandas as pd
import sys
import math

sys.path.append("..//")
from preprocess.raw_data_encode import raw_data_encode


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):

        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        # 定义查询、键和值的线性变换
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.input_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output


class ModelWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(ModelWithAttention, self).__init__()
        self.linear_criteria = nn.Linear(768 * 2, input_dim)
        self.linear_drug = nn.Linear(768, input_dim)
        self.linear_disease = nn.Linear(768, input_dim)
        self.linear_phase = nn.Linear(4, input_dim)
        self.linear_title = nn.Linear(768, input_dim)
        self.linear_summary = nn.Linear(768, input_dim)
        self.linear_primary_purpose = nn.Linear(768, input_dim)
        self.linear_time_frame = nn.Linear(768, input_dim)
        self.linear_intervention_model = nn.Linear(5, input_dim)
        self.linear_masking = nn.Linear(4, input_dim)
        self.linear_enrollment = nn.Linear(1, input_dim)
        self.linear_location = nn.Linear(1, input_dim)
        self.attention_layer = AttentionLayer(input_dim)  # 注意力层
        self.linear1 = nn.Linear(input_dim*5+4+5+4+1+1, input_dim)
        # self.dropout = nn.Dropout(0.1)
        # self.relu = nn.ReLU()
        # self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb,
                primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                enrollment_emb, location_emb):
        criteria_emb = self.linear_criteria(criteria_emb)
        # drug_emb = self.linear_drug(drug_emb)
        # disease_emb = self.linear_disease(disease_emb)
        # phase_emb = self.linear_phase(phase_emb)
        title_emb = self.linear_title(title_emb)
        summary_emb = self.linear_summary(summary_emb)
        # primary_purpose_emb = self.linear_primary_purpose(primary_purpose_emb)
        # time_frame_emb = self.linear_time_frame(time_frame_emb)
        # intervention_model_emb = self.linear_intervention_model(intervention_model_emb)
        # masking_emb = self.linear_masking(masking_emb)
        # enrollment_emb = self.linear_enrollment(enrollment_emb)
        # location_emb = self.linear_location(location_emb)

        embeddings = torch.stack([criteria_emb, title_emb, summary_emb], dim=1)  # (batch_size, P, input_dim)

        attended_emb = self.attention_layer(embeddings)  # (batch_size, P, input_dim)

        output = attended_emb.sum(dim=1)
        output = torch.cat([output, drug_emb, disease_emb, phase_emb, primary_purpose_emb, time_frame_emb,
                            intervention_model_emb, masking_emb, enrollment_emb, location_emb], dim=1)
        output = self.linear1(output)
        # output = self.dropout(output)
        # output = self.relu(output)
        # output = self.linear2(output)

        return output


class Protocol_Embedding_Regression(nn.Sequential):
    def __init__(self, output_dim):
        super(Protocol_Embedding_Regression, self).__init__()
        self.input_dim = 768
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim * 2, output_dim)
        self.f = F.relu

    def forward(self, inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask):
        inclusion_vec = torch.mean(inclusion_emb * inclusion_mask.unsqueeze(-1), dim=1)
        exclusion_vec = torch.mean(exclusion_emb * exclusion_mask.unsqueeze(-1), dim=1)

        protocol_mat = torch.cat([inclusion_vec, exclusion_vec], 1)
        output = self.f(self.fc(protocol_mat))
        return output

    @property
    def embedding_size(self):
        return self.output_dim


class Protocol_Attention_Regression(nn.Module):
    def __init__(self, sentence_embedding_dim, linear_output_dim, transformer_encoder_layers=2, num_heads=6,
                 dropout=0.1, pooling_method='cls'):
        super(Protocol_Attention_Regression, self).__init__()

        # Validate pooling method
        if pooling_method not in ["mean", "max", "cls"]:
            print(f"Invalid pooling method: {pooling_method}. Using 'cls' pooling method.")
            pooling_method = "cls"
        self.pooling_method = pooling_method

        self.cls_token = Parameter(torch.rand(1, 1, sentence_embedding_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=sentence_embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True,
            dim_feedforward=sentence_embedding_dim)
        layer_norm = nn.LayerNorm(sentence_embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=transformer_encoder_layers, norm=layer_norm)

        # self.attetion_l = ModelWithAttention(sentence_embedding_dim)

        # self.linear1 = nn.Linear(3*sentence_embedding_dim+4, sentence_embedding_dim)
        self.linear1 = nn.Linear(8 * sentence_embedding_dim + 5 + 4 + 4 + 2 + 50 + 50,
                                 sentence_embedding_dim)  # 7*sentence_embedding_dim+5+4+4+2
        # self.linear1 = nn.Linear(6*sentence_embedding_dim+5+4+4+2, sentence_embedding_dim)# 6*sentence_embedding_dim+5+4+4+2
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(sentence_embedding_dim, linear_output_dim)

    def forward(self, *args):
        if len(args) == 14:  # 用于run_NN
            criteria_emb = args[0]
            drug_emb = args[1]
            disease_emb = args[2]
            phase_emb = args[3]
            title_emb = args[4]
            summary_emb = args[5]
            primary_purpose_emb = args[6]
            time_frame_emb = args[7]
            intervention_model_emb = args[8]
            masking_emb = args[9]
            enrollment_emb = args[10]
            location_emb = args[11]
            smiles_emb = args[12]
            icd_emb = args[13]

        elif len(args) == 1 and len(args[0]) == 15:  # 用于lr学习
            raw_tuple = args[0]
            criteria_emb = raw_tuple[1]
            drug_emb = raw_tuple[2]
            disease_emb = raw_tuple[3]
            phase_emb = raw_tuple[4]
            title_emb = raw_tuple[5]
            summary_emb = raw_tuple[6]
            primary_purpose_emb = raw_tuple[7]
            time_frame_emb = raw_tuple[8]
            intervention_model_emb = raw_tuple[9]
            masking_emb = raw_tuple[10]
            enrollment_emb = raw_tuple[11]
            location_emb = raw_tuple[12]
            smiles_emb = raw_tuple[13]
            icd_emb = raw_tuple[14]

        elif len(args) == 1 and isinstance(args[0], np.ndarray):  # 用于shap
            raw_np = args[0]
            criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb, \
                primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb, enrollment_emb, \
                location_emb, smiles_emb, icd_emb = raw_data_encode(raw_np)

        else:
            raise ValueError(
                "Invalid input format. Expected either 14 separate embeddings, a tuple with 15 elements, or a numpy array.")

        # criteria_emb = torch.cat((inclusion_emb, exclusion_emb), dim=1)
        # criteria_emb = torch.cat((self.cls_token.expand(criteria_emb.shape[0], -1, -1), criteria_emb), dim=1)
        #
        # criteria_mask = torch.cat((inclusion_mask, exclusion_mask), dim=1)
        # criteria_mask = torch.cat(
        #     (torch.ones(criteria_emb.shape[0], 1, dtype=torch.bool, device=criteria_emb.device), criteria_mask), dim=1)
        # criteria_encoded = self.transformer_encoder(criteria_emb, src_key_padding_mask=criteria_mask)
        #
        # if self.pooling_method == "cls":
        #     pooled_emb = criteria_encoded[:, 0, :]
        # elif self.pooling_method == "max":
        #     pooled_emb = torch.max(criteria_encoded, dim=1)[0]
        # elif self.pooling_method == "mean":
        #     pooled_emb = torch.mean(criteria_encoded, dim=1)
        # else:
        #     raise ValueError("Invalid pooling method")

        protocol_emb = torch.cat((criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb,
                                  primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
                                  enrollment_emb, location_emb, smiles_emb, icd_emb), dim=1)
        # protocol_emb = torch.cat((criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb,
        #                           primary_purpose_emb, time_frame_emb, intervention_model_emb, location_emb), dim=1)

        output = self.linear1(protocol_emb)
        # output = self.attetion_l(criteria_emb, drug_emb, disease_emb, phase_emb, title_emb, summary_emb,
        #                           primary_purpose_emb, time_frame_emb, intervention_model_emb, masking_emb,
        #                           enrollment_emb, location_emb)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)

        if len(args) == 1 and isinstance(args[0], np.ndarray):
            output = output.detach().numpy().flatten()
            output = pd.Series(output, name='outcome')

        return output


class Protocol_Attention_Regression_With_Phases(nn.Module):
    def __init__(self, sentence_embedding_dim, linear_output_dim, transformer_encoder_layers=2, num_heads=6,
                 dropout=0.1, pooling_method='cls'):
        super(Protocol_Attention_Regression_With_Phases, self).__init__()

        # Validate pooling method
        if pooling_method not in ["mean", "max", "cls"]:
            print(f"Invalid pooling method: {pooling_method}. Using 'cls' pooling method.")
            pooling_method = "cls"
        self.pooling_method = pooling_method

        self.cls_token = Parameter(torch.rand(1, 1, sentence_embedding_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=sentence_embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True,
            dim_feedforward=sentence_embedding_dim)
        layer_norm = nn.LayerNorm(sentence_embedding_dim)

        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=transformer_encoder_layers, norm=layer_norm)

        # self.linear1 = nn.Linear(4*sentence_embedding_dim+4, sentence_embedding_dim)
        self.linear1 = nn.Linear(3 * sentence_embedding_dim, sentence_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(sentence_embedding_dim, linear_output_dim)

    def forward(self, inclusion_emb, inclusion_mask, exclusion_emb, exclusion_mask, drug_emb, disease_emb):
        # Expand cls_token to match the batch size
        # inclu_emb = torch.cat((self.cls_token.expand(inclusion_emb.shape[0], -1, -1), inclusion_emb), dim=1)
        # inclu_mask = torch.cat((torch.ones(inclusion_emb.shape[0], 1, dtype=torch.bool, device=inclusion_emb.device), inclusion_mask), dim=1)
        # inclu_encoded = self.transformer_encoder(inclu_emb, src_key_padding_mask=inclu_mask)

        # exclu_emb = torch.cat((self.cls_token.expand(exclusion_emb.shape[0], -1, -1), exclusion_emb), dim=1)
        # exclu_mask = torch.cat((torch.ones(exclusion_emb.shape[0], 1, dtype=torch.bool, device=exclusion_emb.device), exclusion_mask), dim=1)
        # exclu_encoded = self.transformer_encoder(exclu_emb, src_key_padding_mask=exclu_mask)

        criteria_emb = torch.cat((inclusion_emb, exclusion_emb), dim=1)
        criteria_emb = torch.cat((self.cls_token.expand(criteria_emb.shape[0], -1, -1), criteria_emb), dim=1)

        criteria_mask = torch.cat((inclusion_mask, exclusion_mask), dim=1)
        criteria_mask = torch.cat(
            (torch.ones(criteria_emb.shape[0], 1, dtype=torch.bool, device=criteria_emb.device), criteria_mask), dim=1)
        criteria_encoded = self.transformer_encoder(criteria_emb, src_key_padding_mask=criteria_mask)

        # Adjust pooling method handling
        # if self.pooling_method == "cls":
        #     inclu_pooled_emb = inclu_encoded[:, 0, :]
        #     exclu_pooled_emb = exclu_encoded[:, 0, :]
        # elif self.pooling_method == "max":
        #     inclu_pooled_emb = torch.max(inclu_encoded, dim=1)[0]
        #     exclu_pooled_emb = torch.max(exclu_encoded, dim=1)[0]
        # elif self.pooling_method == "mean":
        #     inclu_pooled_emb = torch.mean(inclu_encoded, dim=1)
        #     exclu_pooled_emb = torch.mean(exclu_encoded, dim=1)
        # else:
        #     raise ValueError("Invalid pooling method")

        if self.pooling_method == "cls":
            pooled_emb = criteria_encoded[:, 0, :]
        elif self.pooling_method == "max":
            pooled_emb = torch.max(criteria_encoded, dim=1)[0]
        elif self.pooling_method == "mean":
            pooled_emb = torch.mean(criteria_encoded, dim=1)
        else:
            raise ValueError("Invalid pooling method")

        protocol_emb = torch.cat((pooled_emb, drug_emb, disease_emb), dim=1)
        # protocol_emb = torch.cat((inclu_pooled_emb, exclu_pooled_emb, drug_emb, disease_emb, phase_emb), dim=1)
        output = self.linear1(protocol_emb)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
