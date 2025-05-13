import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

np.random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from schedulers import cyclic_scheduler
from encoders.tabular_numeric_encoder import Tabular_Numeric_Encoder
import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from encoders.tabular_numeric_encoder import Tabular_Numeric_Encoder

class Encoder_Initialization:

    def __init__(self, features_len, logger, device):
        self.features_len = features_len
        self.logger = logger
        self.device = device

    @classmethod
    def weights_init_he(cls, m):
        if isinstance(m, nn.Linear):
            #print(f"Applying He initialization to: {m}")
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            #print(f"Applying custom initialization to BatchNorm: {m}")
            nn.init.ones_(m.weight)  
            nn.init.zeros_(m.bias)

    def init_tabular_numeric_encoder_model_and_optimizer(self, encoder=None):
        if encoder is None:
            encoder = Tabular_Numeric_Encoder(self.features_len).to(self.device)

        if config.tne_init_custom_lr:
            optimizer = optim.AdamW([
                
                # quality-related layers
                {"params": encoder.quality_layer.parameters(), 
                "lr": config.quality_layer_lr, 
                "weight_decay": config.quality_layer_weight_decay,
                "name": "quality_layer"},
                {"params": encoder.quality_high_embedding.parameters(), 
                "lr": config.quality_high_embedding_lr, 
                "weight_decay": config.quality_high_embedding_weight_decay,
                "name": "quality_high_embedding"},
                {"params": encoder.quality_aux_recon_categ_layer.parameters(), 
                "lr": config.quality_aux_lr, 
                "weight_decay": config.quality_aux_weight_decay,
                "name": "quality_aux_recon_categ_layer"},
                {"params": encoder.quality_aux_recon_logit_layer.parameters(), 
                "lr": config.quality_aux_logit_lr, 
                "weight_decay": config.quality_aux_logit_weight_decay,
                "name": "quality_aux_recon_logit_layer"},

                # resonance-related layers
                {"params": encoder.resonance_layer.parameters(), 
                "lr": config.resonance_layer_lr, 
                "weight_decay": config.resonance_layer_weight_decay,
                "name": "resonance_layer"},
                {"params": encoder.resonance_high_embedding.parameters(), 
                "lr": config.resonance_high_embedding_lr, 
                "weight_decay": config.resonance_high_embedding_weight_decay,
                "name": "resonance_high_embedding"},
                {"params": encoder.resonance_aux_recon_layer.parameters(), 
                "lr": config.resonance_aux_lr, 
                "weight_decay": config.resonance_aux_weight_decay,
                "name": "resonance_aux_recon_layer"},

                # tension-related layers
                {"params": encoder.tension_layer.parameters(), 
                "lr": config.tension_layer_lr, 
                "weight_decay": config.tension_layer_weight_decay,
                "name": "tension_layer"},
                {"params": encoder.tension_high_embedding.parameters(), 
                "lr": config.tension_high_embedding_lr, 
                "weight_decay": config.tension_high_embedding_weight_decay,
                "name": "tension_high_embedding"},
                {"params": encoder.tension_aux_recon_layer.parameters(), 
                "lr": config.tension_aux_lr, 
                "weight_decay": config.tension_aux_weight_decay,
                "name": "tension_aux_recon_layer"},

                # longevity-related layers
                {"params": encoder.longevity_layer.parameters(), 
                "lr": config.longevity_layer_lr, 
                "weight_decay": config.longevity_layer_weight_decay,
                "name": "longevity_layer"},
                {"params": encoder.longevity_high_embedding.parameters(), 
                "lr": config.longevity_high_embedding_lr, 
                "weight_decay": config.longevity_high_embedding_weight_decay,
                "name": "longevity_high_embedding"},
                {"params": encoder.longevity_aux_recon_layer.parameters(), 
                "lr": config.longevity_aux_lr, 
                "weight_decay": config.longevity_aux_weight_decay,
                "name": "longevity_aux_recon_layer"},

                # Combined layer
                {"params": encoder.combined_layer.parameters(), 
                "lr": config.combined_layer_lr, 
                "weight_decay": config.combined_layer_weight_decay,
                "name": "combined_layer"}
            ])

        else:
            optimizer = optim.AdamW(encoder.parameters(), 
                                lr=config.tne_general_optimizer_lr,
                                weight_decay=config.tne_general_optimizer_weight_decay)

        return encoder, optimizer

    def get_max_lrs(self):
        return [
            config.max_tne_general_optimizer_lr,
            config.max_quality_layer_lr, 
            config.max_quality_high_embedding_lr, 
            config.max_quality_aux_lr, 
            config.max_quality_aux_logit_lr,
            config.max_resonance_layer_lr, 
            config.max_resonance_high_embedding_lr, 
            config.max_resonance_aux_lr,
            config.tension_layer_lr,  
            config.max_tension_high_embedding_lr, 
            config.tension_aux_lr,
            config.max_longevity_layer_lr, 
            config.longevity_high_embedding_lr, 
            config.max_longevity_aux_lr,
            config.max_attention_layer_lr,
            config.max_combined_layer_lr
        ]

    def init_scheduler(self, optimizer, train_dataloader, num_epochs=None):
        
        if num_epochs is None:
            num_epochs = config.tne_train_epoch
        
        if config.scheduler_type=="CyclicLRWithRestarts":
            scheduler = cyclic_scheduler.CyclicLRWithRestarts(optimizer=optimizer, batch_size=config.tne_batch_size, epoch_size=len(train_dataloader.dataloader.dataset), restart_period=config.cyclicLRWithRestarts_restart_period, t_mult=config.cyclicLRWithRestarts_t_mult, policy=config.cyclicLRWithRestarts_cyclic_policy, min_lr=config.cyclicLRWithRestarts_min_lr, verbose=True)
        else: 
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.get_max_lrs(), #cap lr
            steps_per_epoch=len(train_dataloader),
            epochs=num_epochs,
            anneal_strategy='cos',           
            cycle_momentum=False,            
            div_factor=10, # initial_lr = max_lr / div_factor
            final_div_factor=10000 # final_lr = max_lr / (div_factor * final_div_factor)
        )
            
        return scheduler
    