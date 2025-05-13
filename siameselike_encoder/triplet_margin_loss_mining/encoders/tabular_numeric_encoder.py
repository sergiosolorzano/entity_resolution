import torch
import torch.nn as nn
import numpy as np

import config

np.random.seed(42)
     
class Tabular_Numeric_Encoder(nn.Module):
    def __init__(self, features_len):
        super(Tabular_Numeric_Encoder, self).__init__()
        self.features_len = features_len

        self.quality_layer = nn.Embedding(num_embeddings=config.tne_encoder_quality_num_categories, 
                                         embedding_dim=config.tne_low_level_category_feature_output_dim)
        
        # self.quality_batchnorm = nn.BatchNorm1d(config.tne_low_level_category_feature_output_dim)
        
        self.resonance_layer = nn.Sequential(
            nn.Linear(config.numeric_normd_input_dim, config.tne_low_level_resonance_output_dim),
            nn.ReLU(),
            nn.Dropout(config.tne_resonance_layer_dropout)
            # nn.BatchNorm1d(config.tne_low_level_resonance_output_dim)
        )

        self.tension_layer = nn.Sequential(
            nn.Linear(config.numeric_normd_input_dim, config.tne_low_level_tension_output_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.tne_tension_layer_dropout)
            # nn.BatchNorm1d(config.tne_low_level_tension_output_dim)
        )

        self.longevity_layer = nn.Sequential(
            nn.Linear(config.longevity_normd_input_dim, config.tne_low_level_time_output_dim),
            nn.ReLU(),
            nn.Dropout(config.tne_time_layer_dropout)
            # nn.BatchNorm1d(config.tne_low_level_time_output_dim)
        )

        #High Level Embeddings
        self.quality_high_embedding = nn.Sequential(
            nn.Linear(config.tne_low_level_category_feature_output_dim, config.tne_high_level_all_output_dim),
            nn.ReLU()
        )

        self.resonance_high_embedding = nn.Sequential(
            nn.Linear(config.tne_low_level_resonance_output_dim, config.tne_high_level_all_output_dim),
            nn.ReLU()
        )
        self.tension_high_embedding = nn.Sequential(
            nn.Linear(config.tne_low_level_tension_output_dim, config.tne_high_level_all_output_dim),
            nn.ReLU()
        )
        self.longevity_high_embedding = nn.Sequential(
            nn.Linear(config.tne_low_level_time_output_dim, config.tne_high_level_all_output_dim),
            nn.ReLU()
        )
        
        feature_dim = config.tne_high_level_all_output_dim
        
        #Combined layer
        combined_layer_input_dim = feature_dim * self.features_len
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_layer_input_dim, config.tne_combined_layer_linear_1_output_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.tne_combined_layer_dropout),
            nn.Linear(config.tne_combined_layer_linear_1_output_dim, config.tne_combined_layer_final_output_dim)
        )

        #reconstruction aux heads
        self.quality_aux_recon_categ_layer = nn.Linear(config.tne_low_level_category_feature_output_dim, 1)
        self.quality_aux_recon_logit_layer = nn.Linear(config.tne_low_level_category_feature_output_dim, config.tne_encoder_quality_num_categories)  # Output logits for each class
        self.resonance_aux_recon_layer = nn.Linear(config.tne_low_level_resonance_output_dim, 1)
        self.tension_aux_recon_layer = nn.Linear(config.tne_low_level_tension_output_dim, 1)
        self.longevity_aux_recon_layer = nn.Linear(config.tne_low_level_time_output_dim, 2)

    def forward(self, quality_normd, resonance_normd, tension_normd, longevity_normd):

        #handle tensor with shape [batch_size, 1] to [batch_size]
        if quality_normd.dim() == 2:
            quality_normd = quality_normd[:, 0]  #Use first column
            quality_normd = quality_normd.long()
        
        # low level embeddings
        x_quality = self.quality_layer(quality_normd).squeeze(dim=1)
        # x_quality = self.quality_batchnorm(x_quality)
        x_resonance = self.resonance_layer(resonance_normd)
        x_tension = self.tension_layer(tension_normd)
        x_longevity = self.longevity_layer(longevity_normd)
        
        #high level embeddings
        x_quality_high = self.quality_high_embedding(x_quality)
        x_resonance_high = self.resonance_high_embedding(x_resonance)
        x_tension_high = self.tension_high_embedding(x_tension)
        x_longevity_high = self.longevity_high_embedding(x_longevity)

        x_quality_high = x_quality_high * config.high_embedding_weights["quality"]
        x_resonance_high = x_resonance_high * config.high_embedding_weights["resonance"]
        x_tension_high = x_tension_high * config.high_embedding_weights["tension"]
        x_longevity_high = x_longevity_high * config.high_embedding_weights["longevity"]
        
        #concat and run through combo layer the high level embeddings
        combined_embeddings = torch.cat([x_quality_high, x_resonance_high, x_tension_high, x_longevity_high], dim=1)

        embedding_output = self.combined_layer(combined_embeddings)

        #aux reconstructions from low level embeddings
        quality_recon = self.quality_aux_recon_categ_layer(x_quality)
        quality_logits_recon = self.quality_aux_recon_logit_layer(x_quality)
        resonance_recon = self.resonance_aux_recon_layer(x_resonance)
        tension_recon = self.tension_aux_recon_layer(x_tension)
        longevity_recon = self.longevity_aux_recon_layer(x_longevity)
        longevity_cos_recon = longevity_recon[:, 0].unsqueeze(1)  
        longevity_sin_recon = longevity_recon[:, 1].unsqueeze(1)  

        return (embedding_output, 
                quality_logits_recon, 
                quality_recon,
                resonance_recon, 
                tension_recon, 
                longevity_cos_recon, 
                longevity_sin_recon,
                )