import torch
from torch.utils.data import Dataset

class SingleDataset(Dataset):
    
    def create_dataset(self, input_data_list, numeric_embedding_features, logger, device):
        
        self.numeric_embedding_feature_inputs = numeric_embedding_features
        self.input_data_list = input_data_list
        self.indices = list(range(len(input_data_list)))
        self.logger = logger
        self.device = device
        return self

    def __len__(self):
        return len(self.input_data_list)
    
    def __getitem__(self, idx):

        piano_features_for_model = self.input_data_list[idx]
        
        piano_quality = self.numeric_embedding_feature_inputs.iloc[idx]["quality"]
        piano_resonance = self.numeric_embedding_feature_inputs.iloc[idx]["resonance"]
        piano_tension = self.numeric_embedding_feature_inputs.iloc[idx]["tension"]
        #added cos/sin at constructor
        piano_longevity_cos = self.numeric_embedding_feature_inputs.iloc[idx]["longevity_cos"]
        piano_longevity_sin = self.numeric_embedding_feature_inputs.iloc[idx]["longevity_sin"]
       
        return (piano_features_for_model,
                torch.tensor(piano_quality, dtype=torch.long).to(self.device),
                torch.tensor(piano_resonance, dtype=torch.float32).to(self.device),
                torch.tensor(piano_tension, dtype=torch.float32).to(self.device),
                torch.tensor(piano_longevity_cos, dtype=torch.float32).to(self.device),
                torch.tensor(piano_longevity_sin, dtype=torch.float32).to(self.device)
                )