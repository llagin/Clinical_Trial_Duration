from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":
    if not os.path.exists('data/input_new_features.csv'):
        input_df = pd.read_csv('data/time_prediction_input.csv', sep='\t')
        raw_df = pd.read_csv('data/raw_data.csv', sep=',', dtype={'masking': str, 'intervention_model': str})
        input_df['intervention_model'] = input_df['intervention_model'].apply(lambda x: str(x).zfill(5))
        input_df['masking'] = input_df['masking'].apply(lambda x: str(x).zfill(4))
        columns_to_merge = ['nctid', 'icdcodes', 'smiless']
        missing_columns = [col for col in columns_to_merge if col not in raw_df.columns]
        merged_df = pd.merge(input_df, raw_df[columns_to_merge], on='nctid', how='inner')
        merged_df.to_csv('data/input_new_features.csv', index=False, sep=',')
    else:
        merged_df = pd.read_csv('data/input_new_features.csv',sep=',', dtype={'masking': str, 'intervention_model': str})
    print(merged_df.shape)
