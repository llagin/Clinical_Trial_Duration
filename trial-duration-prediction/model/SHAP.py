import shap
import numpy as np
import pandas as pd
import torch
from models import Protocol_Attention_Regression
from sklearn.preprocessing import OneHotEncoder
from preprocess.raw_data_encode import raw_data_encode
import pickle
import os

if __name__ == '__main__':
    shap.initjs()
    train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t', dtype={'masking': str, 'intervention_model': str})
    input_data = pd.read_csv(f'../data/time_prediction_input.csv', sep='\t', dtype={'masking': str, 'intervention_model': str})
    train_data = train_data.set_index('nctid')
    input_data = input_data.set_index('nctid')
    feature = ['criteria', 'phase', 'drugs', 'diseases', 'title', 'summary', 'enrollment', 'number_of_location',
               'intervention_model', 'masking', 'primary_purpose', 'time_frame', 'smiless', 'icdcodes']
    train_data = train_data[feature]
    input_data = input_data[feature]
    model = Protocol_Attention_Regression(sentence_embedding_dim=768, linear_output_dim=1,
                                          transformer_encoder_layers=2, num_heads=8, dropout=0.1,
                                          pooling_method="cls")

    model.load_state_dict(torch.load(f"./checkpoints/mlp_checkpoint1.pt"))
    # model.load_state_dict(torch.load(f"E:\\task\\大四\\谈\\毕设\\checkpoint_store\\04\\mlp_checkpoint1.pt"))

    model.eval()
    exp_file = '../data/explainer.pkl'
    # if not os.path.exists(exp_file):
    if True:
        LEN1 = 100
        background_data = train_data.iloc[np.random.choice(train_data.shape[0], LEN1, replace=False)]

        # background_data = preprocess_encoder(background_data)
        # print(background_data.head())

        explainer = shap.KernelExplainer(model, background_data)
        with open(exp_file, 'wb') as explainer_file:
            pickle.dump(explainer, explainer_file)
    else:
        with open(exp_file, 'rb') as explainer_file:
            explainer = pickle.load(explainer_file)
    LEN2 = 100
    print("start explain")
    explain_data = input_data.iloc[np.random.choice(input_data.shape[0], LEN2, replace=False)]
    # explain_data = preprocess_encoder(explain_data)

    shap_values = explainer(explain_data, check_additivity=True, gc_collect=True)# 2*12 + 2**11, nsamples=400
    out_file = '../data/shap_values.pkl'

    with open(out_file, 'wb') as shap_file:
        pickle.dump(shap_values, shap_file)

