import h5py
import torch

class EmbeddingCache:
    def __init__(self, file_path='../data/embeddings_cache.h5'):
        self.file_path = file_path
        self.file = h5py.File(file_path, 'r')

    def query(self, nctid):
        if nctid not in self.file:
            raise KeyError(f"{nctid} not found in HDF5 cache.")
        group = self.file[nctid]

        # 顺序取出，逐个返回
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
                print(f"{nctid}未找到！")

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

