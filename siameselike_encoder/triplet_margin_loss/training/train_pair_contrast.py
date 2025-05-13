import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning import losses
from torchmetrics.classification import BinaryPrecision, BinaryRecall, Accuracy
import torchmetrics
from torchinfo import summary
import matplotlib.pyplot as plt
import gc

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CURRENT_DIR)

import config
from encoders.model_loss_classes import TailSuppressedTripletMarginLoss, ContrastiveLoss
from perf_metrics.dict import metrics, accuracy
from utils.logger import Tne_Logger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from evaluation.evaluate_pair_contrast import Evaluation
    from perf_metrics.plots import Plots
    from encoders.model_file_management import Encoder_File_Management
    from encoders.encoder_initialization import Encoder_Initialization
    from encoders.encoder_logging import Encoder_Logs
    from utils.logger import Tne_Logger

np.random.seed(42)

class Train:
    def __init__(self, evaluation_instance: 'Evaluation',
                 plot_instance: 'Plots',
                 model_file_management_instance: 'Encoder_File_Management',
                 model_init_instance: 'Encoder_Initialization',
                 encoder_logging_instance: 'Encoder_Logs',
                 logger: 'Tne_Logger',
                 device: torch.device):
        
        self.evaluation_instance = evaluation_instance
        self.plot_instance = plot_instance
        self.model_file_management_instance = model_file_management_instance
        self.model_init_instance = model_init_instance
        self.encoder_logging_instance = encoder_logging_instance
        self.logger = logger
        self.device = device

    def train_pairs_T_encoder(self, train_dataloader, eval_dataloader, encoder=None, optimizer=None):

        if encoder == None and optimizer == None:
            encoder, optimizer = self.model_init_instance.init_tabular_numeric_encoder_model_and_optimizer()
        else:
            self.logger.info("Encoder and Optimizer already initd")
        
        scheduler = None
        if config.use_scheduler:
            scheduler = self.model_init_instance.init_scheduler(optimizer, train_dataloader)
        
        save_checkpoint_fname = os.path.join(config.TNE_MODEL_DIR, 
                                            config.tabular_numeric_encoder_checkpoint_fname)
        
        
        self.plot_instance.save_encoder_architecture(str(summary(model=encoder)))

        self.train_model(
            encoder,
            train_dataloader,
            eval_dataloader,
            optimizer,
            scheduler,
            save_checkpoint_fname,
            num_epochs=config.tne_train_epoch,
            margin=config.contrastive_margin
        )

    def log_triplet_records(self, triplet_idx, anchor_idx, positive_idx, negative_idx, logger=None):
        log_msg = (f"Training Triplet #{triplet_idx} - indices - Anchor: {anchor_idx}, Positive: {positive_idx}, Negative: {negative_idx}")
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)


    def train_model(self, model, train_dataloader, eval_dataloader, optimizer, scheduler, save_checkpoint_fname, 
                    num_epochs=10, margin=1.0):
        
        model.train()

        loss_feature_dim = config.tne_combined_layer_final_output_dim

        if config.loss_function=="tripletmargin":
            criterion = nn.TripletMarginLoss(config.contrastive_margin, p=2).to(self.device)
        elif config.loss_function == "tripletmargin_mining":
            criterion = TailSuppressedTripletMarginLoss(self.logger, self.device, output_feature_dim=loss_feature_dim, margin=config.contrastive_margin).to(self.device)
        elif config.loss_function=="contrastive":
            criterion = ContrastiveLoss(config.contrastive_margin).to(self.device)
        
        epoch_ap_distances = []
        epoch_an_distances = []
        epoch_total_valid_pos = 0
        epoch_total_valid_neg = 0
        epoch_total_valid_trip = 0
        epoch_total_samples = 0
        trip_percent_history = []

        learned_representation_similarity_anchor_pos_epoch_list = []
        learned_representation_similarity_anchor_neg_epoch_list = []

        total_loss_history = []
        similarity_loss_history = []
        aux_loss_history = []

        self.logger.info("\n*** Running Triplet Training ***\n")

        #save checkpt params
        min_loss = 100
        min_loss_epoch = 0
        quality_computed_accuracy_anchor = 0
        quality_computed_accuracy_pos = 0
        quality_computed_accuracy_neg = 0
        resonance_mse_anchor = 0
        resonance_mse_pos = 0
        resonance_mse_neg = 0
        tension_mse_anchor = 0
        tension_mse_pos = 0
        tension_mse_neg = 0
        longevity_mse_cos_anchor = 0
        longevity_mse_cos_pos = 0
        longevity_mse_cos_neg = 0
        longevity_mse_sin_anchor = 0
        longevity_mse_sin_pos = 0
        longevity_mse_sin_neg = 0

        all_features_recon = []
        
        batch_aux_loss = torch.tensor(0.0, device=self.device)
        batch_triplet_loss = torch.tensor(0.0, device=self.device)
        
        #init metrics
        quality_accuracy_anchor = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)
        quality_accuracy_pos = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)
        quality_accuracy_neg = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)
        precision_metric_anchor_pos = BinaryPrecision().to(self.device)
        precision_metric_anchor_neg = BinaryPrecision().to(self.device)
        precision_metric_pos_neg = BinaryPrecision().to(self.device)
        recall_metric_anchor_pos = BinaryRecall().to(self.device)
        recall_metric_anchor_neg = BinaryRecall().to(self.device)
        recall_metric_pos_neg = BinaryRecall().to(self.device)
        mse_metric_anchor_pos = torchmetrics.MeanSquaredError().to(self.device)
        mse_metric_anchor_neg = torchmetrics.MeanSquaredError().to(self.device)
        mse_metric_pos_neg = torchmetrics.MeanSquaredError().to(self.device)

        model.apply(self.model_init_instance.weights_init_he)

        for epoch in range(num_epochs):

            if config.scheduler_type == "CyclicLRWithRestarts":
                scheduler.step()
            
            total_loss = torch.tensor(0.0, device=self.device)
            similarity_loss = torch.tensor(0.0, device=self.device)
            aux_loss = torch.tensor(0.0, device=self.device)
            
            epoch_total_triplet_loss = torch.tensor(0.0, device=self.device)
            epoch_total_aux_loss = torch.tensor(0.0, device=self.device)

            #reset counter at start epoch
            epoch_total_valid_pos = 0
            epoch_total_valid_neg = 0
            epoch_total_valid_trip = 0
            epoch_total_samples = 0

            #umap collection
            if config.create_emb_maniform:
                epoch_anchor_embeddings = []
                epoch_positive_embeddings = []
                epoch_negative_embeddings = []
            
            #reset metrics
            quality_accuracy_anchor.reset()
            quality_accuracy_pos.reset()
            quality_accuracy_neg.reset()
            precision_metric_anchor_pos.reset()
            precision_metric_anchor_neg.reset()
            precision_metric_pos_neg.reset()
            recall_metric_anchor_pos.reset()
            recall_metric_anchor_neg.reset()
            recall_metric_pos_neg.reset()
            mse_metric_anchor_pos.reset()
            mse_metric_anchor_neg.reset()
            mse_metric_pos_neg.reset()

            for i, (anchor_piano, positive_piano, negative_piano,
                anchor_piano_quality_raw_actual, positive_piano_quality_raw_actual, negative_piano_quality_raw_actual,
                anchor_piano_resonance_raw_actual, positive_piano_resonance_raw_actual, negative_piano_resonance_raw_actual,
                anchor_piano_tension_raw_actual, positive_piano_tension_raw_actual, negative_piano_tension_raw_actual,
                anchor_piano_longevity_cos_raw_actual, positive_piano_longevity_cos_raw_actual, negative_piano_longevity_cos_raw_actual,
                anchor_piano_longevity_sin_raw_actual, positive_piano_longevity_sin_raw_actual, negative_piano_longevity_sin_raw_actual,
                triplet_idx, anchor_piano_idx, positive_piano_idx, negative_piano_idx
                ) in enumerate(train_dataloader):

                if config.log_dataloader_triplet_idx:
                    self.log_triplet_records(
                        triplet_idx=triplet_idx,
                        anchor_idx=anchor_piano_idx,
                        positive_idx=positive_piano_idx,
                        negative_idx=negative_piano_idx,
                        logger=self.logger
                    )

                optimizer.zero_grad()

                #fwd pass
                (anchor_embedding, anchor_quality_logits_recon, anchor_quality_recon, anchor_resonance_recon,
                anchor_tension_recon, anchor_longevity_cos_recon, anchor_longevity_sin_recon)  = model(*anchor_piano)
                
                (positive_embedding, positive_quality_logits_recon, positive_quality_recon, positive_resonance_recon,
                positive_tension_recon, positive_longevity_cos_recon, positive_longevity_sin_recon)  = model(*positive_piano)
                
                (negative_embedding, negative_quality_logits_recon, negative_quality_recon, negative_resonance_recon,
                negative_tension_recon, negative_longevity_cos_recon, negative_longevity_sin_recon)  = model(*negative_piano)
                
                anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                positive_embedding = F.normalize(positive_embedding, p=2, dim=1)
                negative_embedding = F.normalize(negative_embedding, p=2, dim=1)

                anchor_embedding = anchor_embedding.to(self.device)
                positive_embedding = positive_embedding.to(self.device)
                negative_embedding = negative_embedding.to(self.device)

                if config.create_emb_maniform:
                    epoch_anchor_embeddings.append(anchor_embedding.detach().cpu())
                    epoch_positive_embeddings.append(positive_embedding.detach().cpu())
                    epoch_negative_embeddings.append(negative_embedding.detach().cpu())
                
                anchor_quality_logits_recon = anchor_quality_logits_recon.to(self.device)
                positive_quality_logits_recon = positive_quality_logits_recon.to(self.device)
                negative_quality_logits_recon = negative_quality_logits_recon.to(self.device)
                
                anchor_resonance_recon = anchor_resonance_recon.to(self.device)
                positive_resonance_recon = positive_resonance_recon.to(self.device)
                negative_resonance_recon = negative_resonance_recon.to(self.device)
                
                anchor_tension_recon = anchor_tension_recon.to(self.device)
                positive_tension_recon = positive_tension_recon.to(self.device)
                negative_tension_recon = negative_tension_recon.to(self.device)
                
                anchor_longevity_cos_recon = anchor_longevity_cos_recon.to(self.device)
                positive_longevity_cos_recon = positive_longevity_cos_recon.to(self.device)
                negative_longevity_cos_recon = negative_longevity_cos_recon.to(self.device)
                anchor_longevity_sin_recon = anchor_longevity_sin_recon.to(self.device)
                positive_longevity_sin_recon = positive_longevity_sin_recon.to(self.device)
                negative_longevity_sin_recon = negative_longevity_sin_recon.to(self.device)
                
                dist_anchor_pos = torch.norm(anchor_embedding - positive_embedding, p=2, dim=1)
                dist_anchor_neg = torch.norm(anchor_embedding - negative_embedding, p=2, dim=1)
                dist_pos_neg    = torch.norm(positive_embedding - negative_embedding, p=2, dim=1)
                learned_representation_similarity_anchor_pos = torch.clamp(1 - (dist_anchor_pos / margin), min=0, max=1)
                learned_representation_similarity_anchor_neg = torch.clamp(1 - (dist_anchor_neg / margin), min=0, max=1)
                learned_representation_similarity_pos_neg = torch.clamp(1 - (dist_pos_neg / margin), min=0, max=1)

                learned_representation_similarity_anchor_pos_epoch_list.extend(learned_representation_similarity_anchor_pos.cpu().tolist())
                learned_representation_similarity_anchor_neg_epoch_list.extend(learned_representation_similarity_anchor_neg.cpu().tolist())
                
                #aux loss
                quality_aux_anchor_loss = F.cross_entropy(anchor_quality_logits_recon, anchor_piano_quality_raw_actual)
                quality_aux_positive_loss = F.cross_entropy(positive_quality_logits_recon, positive_piano_quality_raw_actual)
                quality_aux_negative_loss = F.cross_entropy(negative_quality_logits_recon, negative_piano_quality_raw_actual)

                resonance_aux_anchor_loss = F.mse_loss(anchor_resonance_recon, anchor_piano_resonance_raw_actual.unsqueeze(1))
                resonance_aux_positive_loss = F.mse_loss(positive_resonance_recon, positive_piano_resonance_raw_actual.unsqueeze(1))
                resonance_aux_negative_loss = F.mse_loss(negative_resonance_recon, negative_piano_resonance_raw_actual.unsqueeze(1))

                tension_aux_anchor_loss = F.mse_loss(anchor_tension_recon, anchor_piano_tension_raw_actual.unsqueeze(1))
                tension_aux_positive_loss = F.mse_loss(positive_tension_recon, positive_piano_tension_raw_actual.unsqueeze(1))
                tension_aux_negative_loss = F.mse_loss(negative_tension_recon, negative_piano_tension_raw_actual.unsqueeze(1))

                longevity_cos_aux_anchor_loss = F.mse_loss(anchor_longevity_cos_recon, anchor_piano_longevity_cos_raw_actual.unsqueeze(1))
                longevity_cos_aux_positive_loss = F.mse_loss(positive_longevity_cos_recon, positive_piano_longevity_cos_raw_actual.unsqueeze(1))
                longevity_cos_aux_negative_loss = F.mse_loss(negative_longevity_cos_recon, negative_piano_longevity_cos_raw_actual.unsqueeze(1))

                longevity_sin_aux_anchor_loss = F.mse_loss(anchor_longevity_sin_recon, anchor_piano_longevity_sin_raw_actual.unsqueeze(1))
                longevity_sin_aux_positive_loss = F.mse_loss(positive_longevity_sin_recon, positive_piano_longevity_sin_raw_actual.unsqueeze(1))
                longevity_sin_aux_negative_loss = F.mse_loss(negative_longevity_sin_recon, negative_piano_longevity_sin_raw_actual.unsqueeze(1))
                
                batch_aux_loss = (quality_aux_anchor_loss + 
                                quality_aux_positive_loss + 
                                quality_aux_negative_loss + 
                                resonance_aux_anchor_loss + 
                                resonance_aux_positive_loss + 
                                resonance_aux_negative_loss + 
                                tension_aux_anchor_loss + 
                                tension_aux_positive_loss + 
                                tension_aux_negative_loss + 
                                longevity_cos_aux_anchor_loss + 
                                longevity_cos_aux_positive_loss + 
                                longevity_cos_aux_negative_loss + 
                                longevity_sin_aux_anchor_loss + 
                                longevity_sin_aux_positive_loss + 
                                longevity_sin_aux_negative_loss
                                )
                epoch_total_aux_loss += batch_aux_loss.detach()
                
                if config.loss_function == "tripletmargin_mining":
                        batch_outlier_triplet_loss, batch_valid_pos, batch_valid_neg, batch_valid_trip, batch_total, dist_ap, dist_an, pos_threshold, neg_threshold  = criterion(
                            anchor_embedding,
                            positive_embedding,
                            negative_embedding,
                            epoch
                        )

                        #accumulate dist
                        epoch_ap_distances.extend(dist_ap.cpu().tolist())
                        epoch_an_distances.extend(dist_an.cpu().tolist())
                        #acccumulate valid samples
                        epoch_total_valid_pos += batch_valid_pos
                        epoch_total_valid_neg += batch_valid_neg
                        epoch_total_valid_trip += batch_valid_trip
                        epoch_total_samples += batch_total

                        batch_triplet_loss = batch_outlier_triplet_loss
                
                elif config.loss_function == "contrastive":
                    loss_ap, _ = criterion(anchor_embedding, positive_embedding, torch.zeros_like(learned_representation_similarity_anchor_pos))
                    loss_an, _ = criterion(anchor_embedding, negative_embedding, torch.ones_like(learned_representation_similarity_anchor_neg))
                    # self.logger.info(f"loss_ap {loss_ap} loss_an {loss_an}")
                    batch_triplet_loss = loss_ap + loss_an

                    #accumulate dist
                    epoch_ap_distances.extend(dist_anchor_pos.cpu().tolist())
                    epoch_an_distances.extend(dist_anchor_neg.cpu().tolist())

                elif config.loss_function == "tripletmargin":
                    
                    batch_triplet_loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

                    #accumulate dist
                    epoch_ap_distances.extend(dist_anchor_pos.cpu().tolist())
                    epoch_an_distances.extend(dist_anchor_neg.cpu().tolist())
                    
                epoch_total_triplet_loss += batch_triplet_loss.detach()

                self.logger.info(f"Batch {i} - Triplet Loss {batch_triplet_loss} Aux Loss {batch_aux_loss}")

                # alpha = min(epoch / num_epochs, 0.5)
                # loss = (alpha * batch_triplet_loss if alpha>0.25 else 0) + ((1-alpha) * batch_aux_loss if alpha>0.25 else batch_aux_loss)
                # loss = (alpha * batch_aux_loss if alpha>0.25 else 0) + ((1-alpha) * batch_triplet_loss if alpha>0.25 else batch_aux_loss)
                loss = batch_triplet_loss + batch_aux_loss
                
                #update classification metrics threshold based
                learned_representation_labels_anchor_pos = (learned_representation_similarity_anchor_pos >= config.contrastive_threshold).float()
                learned_representation_labels_anchor_neg = (learned_representation_similarity_anchor_neg < config.contrastive_threshold).float()
                learned_representation_labels_pos_neg = (learned_representation_similarity_pos_neg < config.contrastive_threshold).float()

                #accuracy count
                accuracy_batch_anchor_pos = (learned_representation_labels_anchor_pos == 1).float().mean()
                accuracy_batch_anchor_neg = (learned_representation_labels_anchor_neg == 1).float().mean()
                accuracy_batch_pos_neg = (learned_representation_labels_pos_neg == 1).float().mean()

                #accuracy quality
                quality_accuracy_anchor.update(anchor_quality_logits_recon.argmax(dim=1), anchor_piano_quality_raw_actual)
                quality_accuracy_pos.update(positive_quality_logits_recon.argmax(dim=1), positive_piano_quality_raw_actual)
                quality_accuracy_neg.update(negative_quality_logits_recon.argmax(dim=1), negative_piano_quality_raw_actual)

                #precision/recall metrics
                precision_metric_anchor_pos.update(learned_representation_labels_anchor_pos, torch.ones(learned_representation_labels_anchor_pos.size(0), device=learned_representation_labels_anchor_pos.device))
                precision_metric_anchor_neg.update(learned_representation_labels_anchor_neg, torch.ones(learned_representation_labels_anchor_neg.size(0), device = learned_representation_labels_anchor_neg.device))
                precision_metric_pos_neg.update(learned_representation_labels_pos_neg, torch.ones(learned_representation_labels_pos_neg.size(0), device = learned_representation_labels_pos_neg.device))
                
                recall_metric_anchor_pos.update(learned_representation_labels_anchor_pos, torch.ones(learned_representation_labels_anchor_pos.size(0), device=learned_representation_labels_anchor_pos.device))
                recall_metric_anchor_neg.update(learned_representation_labels_anchor_neg, torch.ones(learned_representation_labels_anchor_neg.size(0), device=learned_representation_labels_anchor_neg.device))
                recall_metric_pos_neg.update(learned_representation_labels_pos_neg, torch.ones(learned_representation_labels_pos_neg.size(0), device=learned_representation_labels_pos_neg.device))

                #mse metrics
                mse_metric_anchor_pos.update(learned_representation_similarity_anchor_pos, torch.ones(learned_representation_similarity_anchor_pos.size(0), device=learned_representation_similarity_anchor_pos.device))
                mse_metric_anchor_neg.update(learned_representation_similarity_anchor_neg, torch.zeros(learned_representation_similarity_anchor_neg.size(0), device=learned_representation_similarity_anchor_neg.device))
                mse_metric_pos_neg.update(learned_representation_similarity_pos_neg, torch.zeros(learned_representation_similarity_pos_neg.size(0), device=learned_representation_similarity_pos_neg.device))

                loss.backward()

                if config.print_training_weights_grads:
                    self.encoder_logging_instance.report_weights_and_grads(model)

                if config.clip_aux_layers_grad_max_norm > 0:
                    if hasattr(model, 'quality_aux_recon_categ_layer'):
                        #print("Applied clipping to quality_aux_recon_categ_layer")
                        torch.nn.utils.clip_grad_norm_(model.quality_aux_recon_categ_layer.parameters(), config.clip_aux_layers_grad_max_norm)

                    if hasattr(model, 'resonance_aux_recon_layer'):
                        #print("Applied clipping to resonance_aux_recon_layer")
                        torch.nn.utils.clip_grad_norm_(model.resonance_aux_recon_layer.parameters(), config.clip_aux_layers_grad_max_norm)

                    if hasattr(model, 'tension_aux_recon_layer'):
                        #print("Applied clipping to tension_aux_recon_layer")
                        torch.nn.utils.clip_grad_norm_(model.tension_aux_recon_layer.parameters(), config.clip_aux_layers_grad_max_norm)

                    if hasattr(model, 'longevity_aux_recon_layer'):
                        #print("Applied clipping to longevity_aux_recon_layer")
                        torch.nn.utils.clip_grad_norm_(model.longevity_aux_recon_layer.parameters(), config.clip_aux_layers_grad_max_norm)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip_grad_max_norm)

                optimizer.step()

                if config.use_scheduler:
                    if config.scheduler_type=="CyclicLRWithRestarts":
                        scheduler.batch_step()
                    else:
                        scheduler.step()
                
                total_loss += loss.detach()
                similarity_loss += batch_triplet_loss.detach()
                aux_loss += batch_aux_loss.detach()

                if epoch == num_epochs - 1:
                    with torch.no_grad(): 
                        batch_features_recon = pd.DataFrame({
                        "raw_actual_anchor_quality": anchor_piano_quality_raw_actual.cpu().numpy().flatten(),
                        "anchor_quality_logits_recon": torch.argmax(anchor_quality_logits_recon, dim=1).cpu().numpy(),
                        
                        "raw_actual_positive_quality": positive_piano_quality_raw_actual.cpu().numpy().flatten(),
                        "positive_quality_logits_recon": torch.argmax(positive_quality_logits_recon, dim=1).cpu().numpy(),
                        
                        "raw_actual_negative_quality": negative_piano_quality_raw_actual.cpu().numpy().flatten(),
                        "negative_quality_logits_recon": torch.argmax(negative_quality_logits_recon, dim=1).cpu().numpy(),
                        
                        "raw_actual_anchor_resonance": anchor_piano_resonance_raw_actual.cpu().numpy().flatten(),
                        "anchor_resonance_recon": anchor_resonance_recon.cpu().numpy().flatten(),
                        
                        "raw_actual_positive_resonance": positive_piano_resonance_raw_actual.cpu().numpy().flatten(),
                        "positive_resonance_recon": positive_resonance_recon.cpu().numpy().flatten(),

                        "raw_actual_negative_resonance": negative_piano_resonance_raw_actual.cpu().numpy().flatten(),
                        "negative_resonance_recon": negative_resonance_recon.cpu().numpy().flatten(),
                        
                        "raw_actual_anchor_tension": anchor_piano_tension_raw_actual.cpu().numpy().flatten(),
                        "anchor_tension_recon": anchor_tension_recon.cpu().numpy().flatten(),

                        "raw_actual_positive_tension": positive_piano_tension_raw_actual.cpu().numpy().flatten(),
                        "positive_tension_recon": positive_tension_recon.cpu().numpy().flatten(),
                        
                        "raw_actual_negative_tension": negative_piano_tension_raw_actual.cpu().numpy().flatten(),
                        "negative_tension_recon": negative_tension_recon.cpu().numpy().flatten(),
                        
                        "raw_actual_anchor_longevity_cos": anchor_piano_longevity_cos_raw_actual.cpu().numpy().flatten(),
                        "anchor_longevity_cos_recon": anchor_longevity_cos_recon.cpu().numpy().flatten(),

                        "raw_actual_positive_longevity_cos": positive_piano_longevity_cos_raw_actual.cpu().numpy().flatten(),
                        "positive_longevity_cos_recon": positive_longevity_cos_recon.cpu().numpy().flatten(),

                        "raw_actual_negative_longevity_cos": negative_piano_longevity_cos_raw_actual.cpu().numpy().flatten(),
                        "negative_longevity_cos_recon": negative_longevity_cos_recon.cpu().numpy().flatten(),

                        "raw_actual_anchor_longevity_sin": anchor_piano_longevity_sin_raw_actual.cpu().numpy().flatten(),
                        "anchor_longevity_sin_recon": anchor_longevity_sin_recon.cpu().numpy().flatten(),

                        "raw_actual_positive_longevity_sin": positive_piano_longevity_sin_raw_actual.cpu().numpy().flatten(),
                        "positive_longevity_sin_recon": positive_longevity_sin_recon.cpu().numpy().flatten(),

                        "raw_actual_negative_longevity_sin": negative_piano_longevity_sin_raw_actual.cpu().numpy().flatten(),
                        "negative_longevity_sin_recon": negative_longevity_sin_recon.cpu().numpy().flatten(),
                    })
                        all_features_recon.append(batch_features_recon)

            avg_epoch_loss = total_loss / len(train_dataloader)
            total_loss_history.append(avg_epoch_loss.item())
            avg_epoch_triplet_loss = similarity_loss / len(train_dataloader)
            similarity_loss_history.append(avg_epoch_triplet_loss.item())
            avg_epoch_aux_loss = aux_loss / len(train_dataloader)
            aux_loss_history.append(avg_epoch_aux_loss.item())

            if config.scheduler_type == "OneCycleLR":
                if i==0: self.logger.info(f"Batch 0 epoch {epoch} Current OneCycleLR Learning Rate: {scheduler.get_last_lr()} epoch avg cum loss {avg_epoch_loss.item()}")

            if config.loss_function == "tripletmargin_mining":
               
                #plot triplet distribution
                if epoch % config.mid_training_operations_epoch_freq == 0:
                    self.plot_instance.plot_distance_distributions_for_epoch(
                        epoch,
                        epoch_ap_distances,
                        epoch_an_distances,
                        "Train"
                    )

                #plot triplet survival
                trip_percent_history.append(100*epoch_total_valid_trip/epoch_total_samples)
                
                if epoch % config.mid_training_operations_epoch_freq == 0:
                    self.plot_instance.plot_mining_triplet_survived(trip_percent_history)

            if config.loss_function == "contrastive" or config.loss_function == "tripletmargin":
               
                #plot distribution
                if epoch % config.mid_training_operations_epoch_freq == 0:
                    self.plot_instance.plot_distance_distributions_for_epoch(
                        epoch,
                        epoch_ap_distances,
                        epoch_an_distances,
                        "Train"
                    )
            
            #track epoch triplet loss
            avg_epoch_triplet_loss = (epoch_total_triplet_loss / len(train_dataloader)).item()
            #track epoch aux loss
            avg_epoch_aux_loss = (epoch_total_aux_loss / len(train_dataloader)).item()
            
            self.logger.info("\n")
            self.logger.info(f"==> Epoch {epoch} - Avg Triplet Loss {avg_epoch_triplet_loss} Avg Aux Loss {avg_epoch_aux_loss}")

            #compute epoch metrics
            epoch_precision_anchor_pos = precision_metric_anchor_pos.compute()
            epoch_precision_anchor_neg = precision_metric_anchor_neg.compute()
            epoch_precision_pos_neg = precision_metric_pos_neg.compute()
            
            epoch_recall_anchor_pos = recall_metric_anchor_pos.compute()
            epoch_recall_anchor_neg = recall_metric_anchor_neg.compute()
            epoch_recall_pos_neg = recall_metric_pos_neg.compute()

            epoch_mse_anchor_pos = mse_metric_anchor_pos.compute()
            epoch_mse_anchor_neg = mse_metric_anchor_neg.compute()
            epoch_mse_pos_neg = mse_metric_pos_neg.compute()

            quality_computed_accuracy_anchor = quality_accuracy_anchor.compute()
            quality_computed_accuracy_pos = quality_accuracy_pos.compute()
            quality_computed_accuracy_neg = quality_accuracy_neg.compute()

            # if config.use_scheduler:
            #     for param_group in optimizer.param_groups:
            #         self.logger.info(f"Learning Rate - {param_group['name']}: {param_group['lr']:.6f}")

            if avg_epoch_loss < min_loss:
                min_loss = avg_epoch_loss
                min_loss_epoch = epoch
                
                resonance_mse_anchor = resonance_aux_anchor_loss
                resonance_mse_pos = resonance_aux_positive_loss
                resonance_mse_neg = resonance_aux_negative_loss

                tension_mse_anchor = tension_aux_anchor_loss
                tension_mse_pos = tension_aux_positive_loss
                tension_mse_neg = tension_aux_negative_loss

                longevity_mse_cos_anchor = longevity_cos_aux_anchor_loss
                longevity_mse_cos_pos = longevity_cos_aux_positive_loss
                longevity_mse_cos_neg = longevity_cos_aux_negative_loss

                longevity_mse_sin_anchor = longevity_sin_aux_anchor_loss
                longevity_mse_sin_pos = longevity_sin_aux_positive_loss
                longevity_mse_sin_neg = longevity_sin_aux_negative_loss
                
                self.model_file_management_instance.save_T_encoder_checkpoint(model, optimizer, epoch, avg_epoch_loss, 
                                        epoch_precision_anchor_pos, epoch_precision_anchor_neg, epoch_precision_pos_neg,
                                        epoch_recall_anchor_pos, epoch_recall_anchor_neg, epoch_recall_pos_neg,
                                        epoch_mse_anchor_pos, epoch_mse_anchor_neg, epoch_mse_pos_neg,
                                        quality_computed_accuracy_anchor, quality_computed_accuracy_pos, quality_computed_accuracy_neg,
                                        resonance_mse_anchor, resonance_mse_pos, resonance_mse_neg,
                                        tension_mse_anchor, tension_mse_pos, tension_mse_neg,
                                        longevity_mse_cos_anchor, longevity_mse_cos_pos, longevity_mse_cos_neg,
                                        longevity_mse_sin_anchor, longevity_mse_sin_pos, longevity_mse_sin_neg,
                                        save_checkpoint_fname)
            
            self.logger.info("\n")
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.12f} MinLoss: {min_loss:.12f} at {min_loss_epoch}")
            
            #cat for umap
            if config.create_emb_maniform:
                all_anchor_emb = torch.cat(epoch_anchor_embeddings, dim=0)
                all_positive_emb = torch.cat(epoch_positive_embeddings, dim=0)
                all_negative_emb = torch.cat(epoch_negative_embeddings, dim=0)
                self.plot_instance.visualize_embedding_in_manifold(all_anchor_emb, all_positive_emb, all_negative_emb, epoch)
                #conserve mem, clear all
                epoch_anchor_embeddings.clear()
                epoch_positive_embeddings.clear()
                epoch_negative_embeddings.clear()
                total_loss_history.clear()
                similarity_loss_history.clear()
                aux_loss_history.clear()
            
            if epoch % config.mid_training_operations_epoch_freq == 0:
                
                self.logger.info(
                f"Training Count Accuracy Anchor-Pos {accuracy_batch_anchor_pos:.4f}, "
                f"Training Count Accuracy Anchor-Neg {accuracy_batch_anchor_neg:.4f}, "
                f"Training Count Accuracy Pos-Neg {accuracy_batch_pos_neg:.4f}, ")
                
                self.logger.info(
                f"Training quality_computed_accuracy_anchor: {quality_computed_accuracy_anchor:.4f}, "
                f"Training quality_computed_accuracy_pos: {quality_computed_accuracy_pos:.4f}, "
                f"Training quality_computed_accuracy_neg: {quality_computed_accuracy_neg:.4f}, ")
                
                self.logger.info(
                f"Training quality_xentropy_anchor: {quality_aux_anchor_loss:.4f}, "
                f"Training quality_xentropy_positive: {quality_aux_positive_loss:.4f}, "
                f"Training quality_xentropy_negative: {quality_aux_negative_loss:.4f}, ")

                self.logger.info(
                f"Training resonance_mse_anchor: {resonance_mse_anchor:.4f}, "
                f"Training resonance_mse_pos: {resonance_mse_pos:.4f}, "
                f"Training resonance_mse_neg: {resonance_mse_neg:.4f}, ")
                
                self.logger.info(
                f"Training tension_mse_anchor: {tension_mse_anchor:.4f}, "
                f"Training tension_mse_pos: {tension_mse_pos:.4f}, "
                f"Training tension_mse_neg: {tension_mse_neg:.4f}, ")
                
                self.logger.info(
                f"Training longevity_mse_cos_anchor: {longevity_mse_cos_anchor:.4f}, "
                f"Training longevity_mse_cos_pos: {longevity_mse_cos_pos:.4f}, "
                f"Training longevity_mse_cos_neg: {longevity_mse_cos_neg:.4f}, ")
                
                self.logger.info(
                f"Training longevity_mse_sin_anchor: {longevity_mse_sin_anchor:.4f}, "
                f"Training longevity_mse_sin_pos: {longevity_mse_sin_pos:.4f}, "
                f"Training longevity_mse_sin_neg: {longevity_mse_sin_neg:.4f}, ")
                
                self.logger.info(
                f"Training Precision Anchor-Pos: {epoch_precision_anchor_pos:.4f}, Recall: {epoch_recall_anchor_pos:.4f}, "
                f"Training Precision Anchor-Neg: {epoch_precision_anchor_neg:.4f}, Recall: {epoch_recall_anchor_neg:.4f}, "
                f"Training Precision Pos-Neg: {epoch_precision_pos_neg:.4f}, Recall: {epoch_recall_pos_neg:.4f}, ")
                
                self.logger.info(
                f"Training MSE Anchor-Pos: {epoch_mse_anchor_pos:.6f}, "    
                f"Training MSE Anchor-Neg: {epoch_mse_anchor_neg:.6f}, "
                f"Training MSE Pos-Neg: {epoch_mse_pos_neg:.6f}\n")
            
            metrics["epoch"].append(epoch)
            metrics["train_quality_xentropy_anchor"].append(quality_aux_anchor_loss.item())
            metrics["train_quality_xentropy_pos"].append(quality_aux_positive_loss.item())
            metrics["train_quality_xentropy_neg"].append(quality_aux_negative_loss.item())
            metrics["train_resonance_mse_anchor"].append(resonance_aux_anchor_loss.item())
            metrics["train_resonance_mse_pos"].append(resonance_aux_positive_loss.item())
            metrics["train_resonance_mse_neg"].append(resonance_aux_negative_loss.item())
            metrics["train_tension_mse_anchor"].append(tension_aux_anchor_loss.item())
            metrics["train_tension_mse_pos"].append(tension_aux_positive_loss.item())
            metrics["train_tension_mse_neg"].append(tension_aux_negative_loss.item())
            metrics["train_longevity_cos_mse_anchor"].append(longevity_cos_aux_anchor_loss.item())
            metrics["train_longevity_cos_mse_pos"].append(longevity_cos_aux_positive_loss.item())
            metrics["train_longevity_cos_mse_neg"].append(longevity_cos_aux_negative_loss.item())
            metrics["train_longevity_sin_mse_anchor"].append(longevity_sin_aux_anchor_loss.item())
            metrics["train_longevity_sin_mse_pos"].append(longevity_sin_aux_positive_loss.item())
            metrics["train_longevity_sin_mse_neg"].append(longevity_sin_aux_negative_loss.item())

            accuracy["epoch"].append(epoch)
            
            if epoch % config.mid_training_operations_epoch_freq == 0:
                self.evaluation_instance.evaluate_tne(model, eval_dataloader, metrics, epoch)
                model.train()

            if min_loss < config.min_loss_break_training:
                break

            if epoch % config.mid_training_operations_epoch_freq == 0:
                self.plot_instance.plot_training_loss(total_loss_history, similarity_loss_history, aux_loss_history, epoch)

            #clean up
            total_loss = 0
            epoch_total_aux_loss = torch.tensor(0.0, device=self.device)
            epoch_total_triplet_loss = torch.tensor(0.0, device=self.device)
            plt.close('All')

            #store results df
            if epoch % config.mid_training_operations_epoch_freq ==0:
                results_df = pd.DataFrame()

                results_df["Ecludian_Distance_AP"] = epoch_ap_distances
                results_df["Ecludian_Distance_AN"] = epoch_an_distances
                
                results_df["Learned_Representation_Similarity_AP"] = learned_representation_similarity_anchor_pos_epoch_list
                results_df["Learned_Representation_Similarity_AN"] = learned_representation_similarity_anchor_neg_epoch_list
                
                df_results_reset = results_df.reset_index(drop=True)
                df_results_reset.index = df_results_reset.index + 1
                csv_file_path = os.path.join(config.PLOTS_DIR, f"train_sim.csv")
                df_results_reset.to_csv(csv_file_path, index=False)

            epoch_ap_distances.clear()
            epoch_an_distances.clear()
            learned_representation_similarity_anchor_pos_epoch_list.clear()
            learned_representation_similarity_anchor_neg_epoch_list.clear()

            #clear cache: too many logs/rtx5080 firmware issue
            if config.create_emb_maniform:
                torch.cuda.empty_cache()
                gc.collect()
        
        self.logger.info("\n")
        self.logger.info("*** Training Features Reconstruction Comparison  ***\n")
        if all_features_recon:
            reconstructions_df = pd.concat(all_features_recon, ignore_index=True)
            csv_file_path = os.path.join(config.PLOTS_DIR, f"train_reconstructions.csv")
            reconstructions_df.to_csv(csv_file_path, index=False)
            self.logger.info(f"Training Feature Reconstruction Results for Last Epoch saved to {csv_file_path}")
        
        if config.create_emb_maniform:
            self.logger.info("\n")
            self.logger.info(f"Create Embeddings animation")
            self.plot_instance.create_embedding_animation()
        
        self.logger.info("\n")
        self.logger.info(f"End of Training: MinLoss: {min_loss:.12f} at {min_loss_epoch}")

