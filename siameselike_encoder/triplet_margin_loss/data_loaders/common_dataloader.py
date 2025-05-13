import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loaders.cuda_dataloader import CudaDataLoader

import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from features.feature_engineering import Feature_Engineering
    from data_loaders.cuda_dataloader import CudaDataLoader
    from data_loaders.single_dataset import SingleDataset
    from data_loaders.triplet_dataset import TripletDataset
    from utils.logger import Tne_Logger
import config

np.random.seed(42)

class Common_DataLoader:
    def __init__(self, feature_engineering_instance: 'Feature_Engineering',
                 triplet_dataset_instance: 'TripletDataset',
                 single_data_loader_instance: 'SingleDataset',
                 logger: 'Tne_Logger',
                 device: torch.device):
        
        self.feature_engineering_instance = feature_engineering_instance
        self.triplet_dataset_instance = triplet_dataset_instance
        self.single_data_loader_instance = single_data_loader_instance
        self.logger = logger
        self.device = device

    def create_numeric_embed_inputs(self, inputs_config, tne_paths, isTraining):

        self.logger.info(f"\nCreate Dataset Numeric from File: {tne_paths["data_path"]}\n")

        #normalize
        numeric_features_normd, labels_filtered = self.feature_engineering_instance.normalize_numeric_data(tne_paths["data_path"],
                                    inputs_config["numeric_features"], tne_paths["labels_path"], isTraining) 

        dataloader = self.create_triplets_dataset(numeric_features_normd, config.tne_train_eval_dataloader_shuffle,
                                    labels_filtered, config.tne_batch_size)

        return dataloader
    
    def create_input_list_pairs(self, numeric_embedding_features, labels_filtered):
        
        input_data_list = []
        
        longevity_tensor = torch.stack([
            torch.tensor(numeric_embedding_features["longevity_cos"], dtype=torch.float32),
            torch.tensor(numeric_embedding_features["longevity_sin"], dtype=torch.float32)
        ], dim=1).to(self.device)  # Shape: [244, 2]

        for i in range(len(numeric_embedding_features["quality"])):  
            input_data_list.append(( 
                torch.tensor(numeric_embedding_features["quality"][i], dtype=torch.long).unsqueeze(0).to(self.device),
                torch.tensor(numeric_embedding_features["resonance"][i], dtype=torch.float32).unsqueeze(0).to(self.device),
                torch.tensor(numeric_embedding_features["tension"][i], dtype=torch.float32).unsqueeze(0).to(self.device),
                longevity_tensor[i].to(dtype=torch.float32, device=self.device)
            ))

        #load labels
        if not config.run_inference_hierarchical_clustering:
            triplet_index_df = labels_filtered
        else:
            triplet_index_df = None

        return input_data_list, triplet_index_df, numeric_embedding_features
    
    def create_single_dataset(self, numeric_features_normd, dataloader_shuffle,
                              labels_csv_file, batch_size):
            
        input_data_list, triplet_index_df, numeric_embedding_features = self.create_input_list_pairs(numeric_features_normd, 
                                                                                                labels_csv_file)

        # print("input_data_list len",len(input_data_list),"print0",input_data_list[0])
        dataset = self.single_data_loader_instance.create_dataset(input_data_list, numeric_embedding_features, self.logger, self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=dataloader_shuffle)

        cuda_dataloader = CudaDataLoader(dataloader, self.device)
        #CheckAllInCuda(cuda_dataloader)

        return cuda_dataloader

    def create_triplets_dataset(self, numeric_features_normd, dataloader_shuffle, 
                            labels_filtered, batch_size):
        
        input_data_list, triplet_index_df, numeric_embedding_features = self.create_input_list_pairs(numeric_features_normd, 
                                                                                                labels_filtered)

        dataset = self.triplet_dataset_instance.create_triplet(triplet_index_df, input_data_list, numeric_embedding_features, self.logger, self.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=dataloader_shuffle)

        cuda_dataloader = CudaDataLoader(dataloader, self.device)
        #CheckAllInCuda(cuda_dataloader)

        return cuda_dataloader

    def create_pairs_dataset_manager(self, inputs_config, tne_paths, isTraining):
        
        self.logger.info(f"\n***Create Pairs Dataset***")
        
        #create num embeddings
        dataloader= self.create_numeric_embed_inputs(inputs_config, tne_paths, isTraining)
        
        return dataloader