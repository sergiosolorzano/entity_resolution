from features.feature_engineering import Feature_Engineering
from data_loaders.data_path_provider import Data_Path_Provider
from data_loaders.common_dataloader import Common_DataLoader
from inference.inference_engine import Inference_Engine
from perf_metrics.data_metrics import Data_Metrics
from data_loaders.triplet_dataset import TripletDataset
from data_loaders.single_dataset import SingleDataset
from perf_metrics.plots import Plots
from evaluation.evaluate_pair_contrast import Evaluation
from training.train_pair_contrast import Train
from encoders.model_file_management import Encoder_File_Management
from encoders.encoder_initialization import Encoder_Initialization
from encoders.encoder_logging import Encoder_Logs
from perf_metrics.dict import metrics, accuracy
from hierarchical_clustering.hierarch_clust import Hierarchical_Clustering
from utils.logger import Tne_Logger

import config

class ER_Context:
    def __init__(self, device):
        self.device = device
        self.logging_instance = Tne_Logger()
        self.plots_instance = Plots(self.logging_instance, self.device)
        self.data_path_provider_instance = Data_Path_Provider(self.logging_instance)
        self.encoder_logging_instance = Encoder_Logs(self.logging_instance)

        self.model_file_management_instance = Encoder_File_Management(self.logging_instance, self.device)
        self.encoder_init_instance = Encoder_Initialization(len(config.inputs_config["numeric_features"]), self.logging_instance, self.device)
        
        self.feature_engineering_instance = Feature_Engineering(self.logging_instance)
        self.triplet_dataset_instance = TripletDataset()
        self.single_data_loader_instance = SingleDataset()
        self.common_data_loader_instance = Common_DataLoader(self.feature_engineering_instance,
                                                             self.triplet_dataset_instance,
                                                             self.single_data_loader_instance,
                                                             self.logging_instance,
                                                             self.device)
        
        self.evaluation_instance = Evaluation(self.plots_instance, metrics, accuracy, self.logging_instance, self.device)
        
        self.training_instance = Train(self.evaluation_instance,
                                       self.plots_instance,
                                       self.model_file_management_instance,
                                       self.encoder_init_instance,
                                       self.encoder_logging_instance,
                                       self.logging_instance,
                                       self.device)
        
        self.inference_engine_instance = Inference_Engine(self.feature_engineering_instance,
                                                          self.common_data_loader_instance,
                                                          self.single_data_loader_instance,
                                                          self.model_file_management_instance,
                                                          self.encoder_init_instance,
                                                          self.logging_instance,
                                                          self.device)
        
        self.data_metrics_instance = Data_Metrics(self.feature_engineering_instance, self.plots_instance,
                                                  self.logging_instance)
        
        self.hierarchical_clustering_instance = Hierarchical_Clustering(self.logging_instance, 
                                                                        self.device)