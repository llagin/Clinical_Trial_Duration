import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from model import Protocol_Attention_Regression_FACT_new
from embeddings_cache import EmbeddingCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Protocol_Attention_Regression_FACT_new(output_dim=1).to(device)
model.load_state_dict(torch.load(
    "checkpoints/mlp_checkpoint_15_21.pt", map_location=device, weights_only=True
))
model.eval()

cache = EmbeddingCache()

input_data_df = pd.read_csv("../data/new_input.csv", sep=',', dtype={'masking': str, 'intervention_model': str})
input_data_df['start_year'] = input_data_df['start_date'].str[-4:]
input_data_df['completion_year'] = input_data_df['completion_date'].str[-4:]

test_df = input_data_df[input_data_df['start_year'] >= '2021']

# Captum
ig = IntegratedGradients(model)

modal_names = [
    "criteria", "drug", "disease", "phase", "title", "summary",
    "primary_purpose", "time_frame", "intervention_model", "masking",
    "enrollment", "location", "smiles", "icd"
]

all_attributions = []

for idx, (_, row) in enumerate(test_df.iterrows()):
    input_list = cache.query_df(pd.DataFrame([row]))
    input_list = [x.to(device) for x in input_list]

    input_tuple = (
        input_list[0], input_list[2], input_list[3], input_list[1],
        input_list[4], input_list[5], input_list[6], input_list[7],
        input_list[8], input_list[9], input_list[10], input_list[11],
        input_list[12], input_list[13]
    )

    baseline_tuple = tuple(torch.zeros_like(x).to(device) for x in input_tuple)

    attributions = ig.attribute(
        inputs=input_tuple,
        baselines=baseline_tuple,
        target=0,
        n_steps=50
    )

    attr_values = np.array([a[0].detach().cpu().sum().item() for a in attributions])
    all_attributions.append(attr_values)

    if (idx + 1) % 20 == 0 or (idx + 1) == len(test_df):
        print(f"Processed {idx + 1}/{len(test_df)} samples")

all_attributions = np.array(all_attributions)

# compute global attribution
mean_attr = np.mean(all_attributions, axis=0)
std_attr = np.std(all_attributions, axis=0)

sorted_idx = np.argsort(mean_attr)
sorted_names = np.array(modal_names)[sorted_idx]
sorted_values = mean_attr[sorted_idx]
sorted_std = std_attr[sorted_idx]

# draw bar plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_names, sorted_values, xerr=sorted_std, color='skyblue')
plt.xlabel('Average Attribution')
plt.title('Global Feature Attribution on Test Set')
plt.tight_layout()
plt.savefig("global_attribution_bar.png", dpi=300, bbox_inches='tight')
plt.show()

# print global attribution
print("\n======== Global Attribution Results ========")
for name, val, std in zip(sorted_names, sorted_values, sorted_std):
    print(f"Feature: {name:20s} | Mean Attribution: {val:.6f} | Std: {std:.6f}")
