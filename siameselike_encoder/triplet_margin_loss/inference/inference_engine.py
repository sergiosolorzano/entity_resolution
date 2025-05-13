import os, sys
import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from features.feature_engineering import Feature_Engineering
    from data_loaders.common_dataloader import Common_DataLoader
    from data_loaders.single_dataset import SingleDataset
    from encoders.model_file_management import Encoder_File_Management
    from encoders.encoder_initialization import Encoder_Initialization
    from utils.logger import Tne_Logger

class Inference_Engine:
    def __init__(self, feature_engineering_instance: 'Feature_Engineering',
                 common_data_loader_instance: 'Common_DataLoader',
                 single_data_loader_instance: 'SingleDataset',
                 model_file_management_instance: 'Encoder_File_Management',
                 model_init_instance: 'Encoder_Initialization',
                 logger: 'Tne_Logger',
                 device: torch.device):
        
        self.feature_engineering_instance = feature_engineering_instance
        self.common_data_loader_instance = common_data_loader_instance
        self.single_data_loader_instance = single_data_loader_instance
        self.model_file_management_instance = model_file_management_instance
        self.model_init_instance = model_init_instance
        self.logger = logger
        self.device = device

    def run_inference(self, provider, inputs_config, tne_train_paths):
        if config.run_inference_hierarchical_clustering:
            
            self.logger.info("\n***Running Inference***\n")
            
            encoder, optimizer = self.model_init_instance.init_tabular_numeric_encoder_model_and_optimizer()
            checkpoint_fpath = os.path.join(config.TNE_MODEL_DIR, config.tabular_numeric_encoder_checkpoint_fname)
            encoder, optimizer, epoch, loss = self.model_file_management_instance.load_T_encoder_checkpoint(checkpoint_fpath, encoder, optimizer)
            self.logger.info(f"Loaded model {checkpoint_fpath}")

            tne_infer_paths = provider.get_paths(mode='tne_mode', is_training=False)

            all_embeddings = self.infer_embeddings_from_tabular_numeric_encoder(tne_infer_paths["data_path"], inputs_config["numeric_features"], encoder, False)

            return all_embeddings

    def infer_embeddings_from_tabular_numeric_encoder(self, csv_file_path, numeric_feature_dict, encoder, isTraining):
        
        numeric_features_normd, labels_filtered = self.feature_engineering_instance.normalize_numeric_data(csv_file_path, numeric_feature_dict, None, isTraining)    
        infer_dataloader = self.common_data_loader_instance.create_single_dataset(numeric_features_normd, False,
                                None, config.tne_batch_size)
        
        encoder.eval()
        
        all_embeddings = []
        for c, (piano_batch, batch_quality_actual, batch_resonance_actual, batch_tension_actual, batch_longevity_cos_actual, batch_longevity_sin_actual) in enumerate(infer_dataloader):
            #reshape and to gpu for debug
            quality_normd = batch_quality_actual.float().unsqueeze(1).to(self.device)
            resonance_normd = batch_resonance_actual.float().unsqueeze(1).to(self.device)
            tension_normd = batch_tension_actual.float().unsqueeze(1).to(self.device)
            longevity_normd = torch.stack([batch_longevity_cos_actual, batch_longevity_sin_actual], dim=1).to(self.device)

            with torch.no_grad():
                embeddings, quality_recon_logits, quality_recon, resonance_recon, tension_recon, longevity_cos_recon, longevity_sin_recon = encoder(*piano_batch)
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            
            quality_aux_anchor_recon_error = F.cross_entropy(quality_recon_logits, batch_quality_actual)
            resonance_aux_negative_recon_error = F.mse_loss(resonance_recon, batch_resonance_actual.unsqueeze(1))
            tension_aux_negative_recon_error = F.mse_loss(tension_recon, batch_tension_actual.unsqueeze(1))
            longevity_cos_aux_negative_recon_error = F.mse_loss(longevity_cos_recon, batch_longevity_cos_actual.unsqueeze(1))
            longevity_sin_aux_negative_recon_error = F.mse_loss(longevity_sin_recon, batch_longevity_sin_actual.unsqueeze(1))

            self.logger.info(f"quality_error {quality_aux_anchor_recon_error}")
            self.logger.info(f"resonance_error {resonance_aux_negative_recon_error}")
            self.logger.info(f"tension_error {tension_aux_negative_recon_error}")
            self.logger.info(f"longevity_error {longevity_cos_aux_negative_recon_error}")
            self.logger.info(f"longevity_error {longevity_sin_aux_negative_recon_error}")
            
            self.logger.info("Inference Feature Reconstruction Results:")
            # self.logger.info(f"Quality Actual vs Reconstructed: {batch_quality_actual} vs {np.mean(quality_recon)}")
            self.logger.info(f"Resonance Actual vs Reconstructed: {batch_resonance_actual.mean().item()} vs {resonance_recon.mean().item()}")
            self.logger.info(f"Tension Actual vs Reconstructed: {batch_tension_actual.mean().item()} vs {tension_recon.mean().item()}")
            self.logger.info(f"Longevity_Cos Actual vs Reconstructed: {batch_longevity_cos_actual.mean().item()} vs {longevity_cos_recon.mean().item()}")
            self.logger.info(f"Longevity_Sin Actual vs Reconstructed: {batch_longevity_sin_actual.mean().item()} vs {longevity_sin_recon.mean().item()}")


        return all_embeddings