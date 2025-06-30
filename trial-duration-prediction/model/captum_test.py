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
explain_df = test_df.sample(n=1, random_state=42)

input_list = cache.query_df(explain_df)
input_list = [x.to(device) for x in input_list]

input_tuple = (
    input_list[0],   # criteria
    input_list[2],   # drug
    input_list[3],   # disease
    input_list[1],   # phase
    input_list[4],   # title
    input_list[5],   # summary
    input_list[6],   # primary_purpose
    input_list[7],   # time_frame
    input_list[8],   # intervention_model
    input_list[9],   # masking
    input_list[10],  # enrollment
    input_list[11],  # location
    input_list[12],  # smiles
    input_list[13]   # icd
)

baseline_tuple = tuple(torch.zeros_like(x).to(device) for x in input_tuple)

# Captum
ig = IntegratedGradients(model)

attributions = ig.attribute(
    inputs=input_tuple,
    baselines=baseline_tuple,
    target=0,
    n_steps=50
)

output = model(*input_tuple).detach().cpu().numpy()[0][0]

# waterfall plot
modal_names = [
    "criteria", "drug", "disease", "phase", "title", "summary",
    "primary_purpose", "time_frame", "intervention_model", "masking",
    "enrollment", "location", "smiles", "icd"
]

# the attribution of single sample
attr_values = np.array([a[0].detach().cpu().sum().item() for a in attributions])

sorted_idx = np.argsort(attr_values)
sorted_names = np.array(modal_names)[sorted_idx]
sorted_values = attr_values[sorted_idx]

cumulative = np.cumsum(np.insert(sorted_values, 0, 0))

plt.figure(figsize=(10, 6))
for i in range(len(sorted_values)):
    color = 'red' if sorted_values[i] < 0 else 'green'
    plt.bar(i, sorted_values[i], bottom=cumulative[i], color=color)

plt.plot(range(len(cumulative)), cumulative, marker='o', color='blue', label='Cumulative')
plt.axhline(y=output, color='black', linestyle='--', label=f'Model Output: {output:.2f}')

plt.xticks(range(len(sorted_names)), sorted_names, rotation=90)
plt.ylabel('Contribution to Output')
plt.title('Feature Attribution')
plt.legend()
plt.tight_layout()
plt.savefig("captum_waterfall_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# print attribution results
print("\n======== Attribution Results ========")
for name, val in zip(modal_names, attr_values):
    print(f"Feature: {name:20s} | Attribution Value: {val:.6f}")
print(f"\nModel Output: {output:.6f}")
