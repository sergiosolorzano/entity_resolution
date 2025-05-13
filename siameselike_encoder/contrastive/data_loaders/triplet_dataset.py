import torch
from torch.utils.data import Dataset
import pandas as pd

class TripletDataset(Dataset):

    def create_triplet(self, triplet_index_df: pd.DataFrame, input_data_list, numeric_embedding_features, logger, device):
        self.triplet_indices = [(row['anchor_piano'], row['positive_piano'], row['negative_piano']) 
                            for _, row in triplet_index_df.iterrows()]
        self.triplet_index_df = triplet_index_df
        self.numeric_embedding_feature_inputs = numeric_embedding_features
        self.input_data_list = input_data_list
        self.logger = logger
        self.device = device
        return self

    def __len__(self):
        return len(self.triplet_index_df)
    
    def __getitem__(self, idx):
        anchor_piano_idx, positive_piano_idx, negative_piano_idx = self.triplet_indices[idx]
        
        anchor_piano_features_for_model = self.input_data_list[anchor_piano_idx]
        positive_piano_features_for_model = self.input_data_list[positive_piano_idx]
        negative_piano_features_for_model = self.input_data_list[negative_piano_idx]
        
        anchor_piano_quality = self.numeric_embedding_feature_inputs.iloc[anchor_piano_idx]["quality"]
        positive_piano_quality = self.numeric_embedding_feature_inputs.iloc[positive_piano_idx]["quality"]
        negative_piano_quality = self.numeric_embedding_feature_inputs.iloc[negative_piano_idx]["quality"]

        anchor_piano_resonance = self.numeric_embedding_feature_inputs.iloc[anchor_piano_idx]["resonance"]
        positive_piano_resonance = self.numeric_embedding_feature_inputs.iloc[positive_piano_idx]["resonance"]
        negative_piano_resonance = self.numeric_embedding_feature_inputs.iloc[negative_piano_idx]["resonance"]

        anchor_piano_tension = self.numeric_embedding_feature_inputs.iloc[anchor_piano_idx]["tension"]
        positive_piano_tension = self.numeric_embedding_feature_inputs.iloc[positive_piano_idx]["tension"]
        negative_piano_tension = self.numeric_embedding_feature_inputs.iloc[negative_piano_idx]["tension"]
        
        # added cos/sin at constructor
        anchor_piano_longevity_cos = self.numeric_embedding_feature_inputs.iloc[anchor_piano_idx]["longevity_cos"]
        positive_piano_longevity_cos = self.numeric_embedding_feature_inputs.iloc[positive_piano_idx]["longevity_cos"]
        negative_piano_longevity_cos = self.numeric_embedding_feature_inputs.iloc[negative_piano_idx]["longevity_cos"]
        
        anchor_piano_longevity_sin = self.numeric_embedding_feature_inputs.iloc[anchor_piano_idx]["longevity_sin"]
        positive_piano_longevity_sin = self.numeric_embedding_feature_inputs.iloc[positive_piano_idx]["longevity_sin"]
        negative_piano_longevity_sin = self.numeric_embedding_feature_inputs.iloc[negative_piano_idx]["longevity_sin"]
       
        anchor_piano_features_for_model = [t.to(self.device) for t in anchor_piano_features_for_model]
        positive_piano_features_for_model = [t.to(self.device) for t in positive_piano_features_for_model]
        negative_piano_features_for_model = [t.to(self.device) for t in negative_piano_features_for_model]
        
        return (anchor_piano_features_for_model, positive_piano_features_for_model, negative_piano_features_for_model,
                torch.tensor(anchor_piano_quality, dtype=torch.long).to(self.device),
                torch.tensor(positive_piano_quality, dtype=torch.long).to(self.device),
                torch.tensor(negative_piano_quality, dtype=torch.long).to(self.device),
                torch.tensor(anchor_piano_resonance, dtype=torch.float32).to(self.device),
                torch.tensor(positive_piano_resonance, dtype=torch.float32).to(self.device),
                torch.tensor(negative_piano_resonance, dtype=torch.float32).to(self.device),
                torch.tensor(anchor_piano_tension, dtype=torch.float32).to(self.device),
                torch.tensor(positive_piano_tension, dtype=torch.float32).to(self.device),
                torch.tensor(negative_piano_tension, dtype=torch.float32).to(self.device),
                torch.tensor(anchor_piano_longevity_cos, dtype=torch.float32).to(self.device),
                torch.tensor(positive_piano_longevity_cos, dtype=torch.float32).to(self.device),
                torch.tensor(negative_piano_longevity_cos, dtype=torch.float32).to(self.device),
                torch.tensor(anchor_piano_longevity_sin, dtype=torch.float32).to(self.device),
                torch.tensor(positive_piano_longevity_sin, dtype=torch.float32).to(self.device),
                torch.tensor(negative_piano_longevity_sin, dtype=torch.float32).to(self.device),
                torch.tensor(idx), torch.tensor(anchor_piano_idx), torch.tensor(positive_piano_idx), torch.tensor(negative_piano_idx)
                )
  