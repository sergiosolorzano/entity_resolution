import os
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, Accuracy
import torchmetrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from perf_metrics.plots import Plots
    from utils.logger import Tne_Logger

np.random.seed(42)

class Evaluation:

    def __init__(self, plot_instance: 'Plots',
                 metrics: dict, accuracy: dict,
                 logger: 'Tne_Logger',
                 device: torch.device):
        
        self.logger = logger
        self.device = device
        self.plot_instance = plot_instance
        
        self.LOSS_FUNCTION_MAPPING = {
            "tripletmargin": lambda model, dataloader, train_metrics, epoch: self.eval_contrastive(
                model, dataloader, config.contrastive_margin, train_metrics, epoch
            ),
            "tripletmargin_mining": lambda model, dataloader, train_metrics, epoch: self.eval_contrastive(
                model, dataloader, config.contrastive_margin, train_metrics, epoch
            ),
            "contrastive": lambda model, dataloader, train_metrics, epoch: self.eval_contrastive(
                model, dataloader, config.contrastive_margin, train_metrics, epoch
            ),
        }
        self.metrics = metrics
        self.accuracy = accuracy

    def evaluate_tne(self, tne_encoder, eval_dataloader, train_metrics, epoch):

        self.logger.info("\n*** Running Evaluation TNE ***\n")

        eval_function = self.LOSS_FUNCTION_MAPPING.get(config.loss_function)
        if eval_function is None:
            raise ValueError(f"Unsupported loss function: {config.loss_function}")
        return eval_function(tne_encoder, eval_dataloader, train_metrics, epoch)

    def initialize_metrics(self):
        quality_accuracy_anchor = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)
        quality_accuracy_pos = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)
        quality_accuracy_neg = Accuracy(task="multiclass", num_classes=config.tne_encoder_quality_num_categories).to(self.device)

        combined_auroc_metric = BinaryAUROC().to(self.device)
        combined_auroc_score = 0
        
        precision_metric_anchor_pos = BinaryPrecision().to(self.device)
        precision_metric_anchor_neg = BinaryPrecision().to(self.device)
        precision_metric_pos_neg = BinaryPrecision().to(self.device)

        recall_metric_anchor_pos = BinaryRecall().to(self.device)
        recall_metric_anchor_neg = BinaryRecall().to(self.device)
        recall_metric_pos_neg = BinaryRecall().to(self.device)

        mse_metric_anchor_pos = torchmetrics.MeanSquaredError().to(self.device)
        mse_metric_anchor_neg = torchmetrics.MeanSquaredError().to(self.device)
        mse_metric_pos_neg = torchmetrics.MeanSquaredError().to(self.device)

        return (
            quality_accuracy_anchor, quality_accuracy_pos, quality_accuracy_neg,
            combined_auroc_metric, combined_auroc_score,
            precision_metric_anchor_pos, precision_metric_anchor_neg, precision_metric_pos_neg,
            recall_metric_anchor_pos, recall_metric_anchor_neg, recall_metric_pos_neg,
            mse_metric_anchor_pos, mse_metric_anchor_neg, mse_metric_pos_neg
        )

    def validate_similarity_ranges(self, learned_representation_sim_anchor_pos, learned_representation_sim_anchor_neg, learned_representation_sim_pos_neg):
        if not torch.all((learned_representation_sim_anchor_pos >= 0) & (learned_representation_sim_anchor_pos <= 1)):
            self.logger.info("WARNING - learned_representation_sim_anchor_pos out of [0,1].")
        if not torch.all((learned_representation_sim_anchor_neg >= 0) & (learned_representation_sim_anchor_neg <= 1)):
            self.logger.info("WARNING - learned_representation_sim_anchor_neg out of [0,1].")
        if not torch.all((learned_representation_sim_pos_neg >= 0) & (learned_representation_sim_pos_neg <= 1)):
            self.logger.info("WARNING - learned_representation_sim_pos_neg out of [0,1].")

    def print_final_results(self,
        df_results_reset, accuracy_anchor, accuracy_pos, accuracy_neg,
        combined_auroc_metric, combined_auroc_score,
        final_precision_anchor_pos, final_precision_anchor_neg, final_precision_pos_neg,
        final_recall_anchor_pos, final_recall_anchor_neg, final_recall_pos_neg,
        final_mse_anchor_pos, final_mse_anchor_neg, final_mse_pos_neg,
        quality_accuracy_anchor, quality_accuracy_pos, quality_accuracy_neg,
        avg_resonance_aux_anchor_loss, avg_resonance_aux_positive_loss, avg_resonance_aux_negative_loss,
        avg_tension_aux_anchor_loss, avg_tension_aux_positive_loss, avg_tension_aux_negative_loss,
        avg_longevity_cos_aux_anchor_loss, avg_longevity_cos_aux_positive_loss, avg_longevity_cos_aux_negative_loss,
        avg_longevity_sin_aux_anchor_loss, avg_longevity_sin_aux_positive_loss, avg_longevity_sin_aux_negative_loss,
        dist_anchor_pos_tensor, dist_anchor_neg_tensor, dist_pos_neg_tensor,
        all_learned_representations_sim_anchor_pos, all_learned_representations_sim_anchor_neg, all_learned_representations_sim_pos_neg,
        avg_quality_aux_anchor_loss, avg_quality_aux_positive_loss, avg_quality_aux_negative_loss
    ):
        self.logger.info("*** Evaluation Accuracy Detailed Results  ***\n")
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', '{:,.6f}'.format):
        #     print(df_reset.head())
        csv_file_path = os.path.join(config.PLOTS_DIR, f"eval_metrics.csv")
        df_results_reset.to_csv(csv_file_path, index=False)
        self.logger.info(f"Evaluation Accuracy Detailed saved to {csv_file_path}")

        self.logger.info(f"Eval Accuracy Anchor: {accuracy_anchor:.2f}%")
        self.logger.info(f"Eval Accuracy Positive: {accuracy_pos:.2f}%")
        self.logger.info(f"Eval Accuracy Negative: {accuracy_neg:.2f}%")

        self.logger.info("\n== Evaluation Reconstruction Metric Results ==")
        self.logger.info(
            f"Combined AUROC Score={combined_auroc_score:.4f}, "
            f"Precision AP={final_precision_anchor_pos:.4f}, Recall AP={final_recall_anchor_pos:.4f}, "
            f"Precision AN={final_precision_anchor_neg:.4f}, Recall AN={final_recall_anchor_neg:.4f}, "
            f"Precision PN={final_precision_pos_neg:.4f}, Recall PN={final_recall_pos_neg:.4f}, "
            f"MSE AP={final_mse_anchor_pos:.4f}, "
            f"MSE AN={final_mse_anchor_neg:.4f}, "
            f"MSE PN={final_mse_pos_neg:.4f}"
        )

        self.logger.info("\n== Eval Aux Loss Results ==")
        self.logger.info(f"Eval quality Accuracy Anchor: {quality_accuracy_anchor.compute():.4f}")
        self.logger.info(f"Eval quality Accuracy Positive: {quality_accuracy_pos.compute():.4f}")
        self.logger.info(f"Eval quality Accuracy Negative: {quality_accuracy_neg.compute():.4f}")

        self.logger.info(f"Eval quality xEntropy Anchor (avg): {avg_quality_aux_anchor_loss:.4f}")
        self.logger.info(f"Eval quality xEntropy Positive (avg): {avg_quality_aux_positive_loss:.4f}")
        self.logger.info(f"Eval quality xEntropy Negative (avg): {avg_quality_aux_negative_loss:.4f}")

        self.logger.info(f"Eval resonance MSE Anchor (avg): {avg_resonance_aux_anchor_loss:.4f}")
        self.logger.info(f"Eval resonance MSE Positive (avg): {avg_resonance_aux_positive_loss:.4f}")
        self.logger.info(f"Eval resonance MSE Negative (avg): {avg_resonance_aux_negative_loss:.4f}")

        self.logger.info(f"Eval tension MSE Anchor (avg): {avg_tension_aux_anchor_loss:.4f}")
        self.logger.info(f"Eval tension MSE Positive (avg): {avg_tension_aux_positive_loss:.4f}")
        self.logger.info(f"Eval tension MSE Negative (avg): {avg_tension_aux_negative_loss:.4f}")

        self.logger.info(f"Eval longevity_cos MSE Anchor (avg): {avg_longevity_cos_aux_anchor_loss:.4f}")
        self.logger.info(f"Eval longevity_cos MSE Positive (avg): {avg_longevity_cos_aux_positive_loss:.4f}")
        self.logger.info(f"Eval longevity_cos MSE Negative (avg): {avg_longevity_cos_aux_negative_loss:.4f}")

        self.logger.info(f"Eval longevity_sin MSE Anchor (avg): {avg_longevity_sin_aux_anchor_loss:.4f}")
        self.logger.info(f"Eval longevity_sin MSE Positive (avg): {avg_longevity_sin_aux_positive_loss:.4f}")
        self.logger.info(f"Eval longevity_sin MSE Negative (avg): {avg_longevity_sin_aux_negative_loss:.4f}")

        self.logger.info(f"Mean Distance AP: {dist_anchor_pos_tensor.mean().item():.4f}")
        self.logger.info(f"Mean Distance AN: {dist_anchor_neg_tensor.mean().item():.4f}")
        self.logger.info(f"Mean Distance PN: {dist_pos_neg_tensor.mean().item():.4f}")

        self.logger.info(f"Mean Learned Representation Similarity AP: {all_learned_representations_sim_anchor_pos.mean().item():.4f}")
        self.logger.info(f"Mean Learned Representation Similarity AN: {all_learned_representations_sim_anchor_neg.mean().item():.4f}")
        self.logger.info(f"Mean Learned Representation Similarity PN: {all_learned_representations_sim_pos_neg.mean().item():.4f}")

    def log_triplet_records(self, triplet_idx, anchor_idx, positive_idx, negative_idx, logger=None):
        log_msg = (f"Evaluation Triplet #{triplet_idx} - indices - Anchor: {anchor_idx}, Positive: {positive_idx}, Negative: {negative_idx} | ")
        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

    def eval_contrastive(self, model, dataloader, margin, train_metrics, epoch):
        
        self.logger.info("*** Evaluation Starts ***")

        with torch.no_grad():
            model.eval()

        (
            quality_accuracy_anchor, quality_accuracy_pos, quality_accuracy_neg,
            combined_auroc_metric,combined_auroc_score,
            precision_metric_anchor_pos, precision_metric_anchor_neg, precision_metric_pos_neg,
            recall_metric_anchor_pos, recall_metric_anchor_neg, recall_metric_pos_neg,
            mse_metric_anchor_pos, mse_metric_anchor_neg, mse_metric_pos_neg
        ) = self.initialize_metrics()

        dist_anchor_pos_list = []
        dist_anchor_neg_list = []
        dist_pos_neg_list = []
        scaled_sim_anchor_pos_list = []
        scaled_sim_anchor_neg_list = []
        scaled_sim_pos_neg_list = []

        learned_representation_sim_anchor_pos_list = []
        learned_representation_sim_anchor_neg_list = []
        learned_representation_sim_pos_neg_list = []
        correct_anchor_list = []
        correct_pos_list = []
        correct_neg_list = []

        all_features_reconstruction_dfs = []

        total_correct_anchor = 0
        total_correct_pos = 0
        total_correct_neg = 0
        num_samples = 0

        total_quality_aux_anchor_loss = 0.0
        total_quality_aux_positive_loss = 0.0
        total_quality_aux_negative_loss = 0.0

        total_resonance_aux_anchor_loss = 0.0
        total_resonance_aux_positive_loss = 0.0
        total_resonance_aux_negative_loss = 0.0

        total_tension_aux_anchor_loss = 0.0
        total_tension_aux_positive_loss = 0.0
        total_tension_aux_negative_loss = 0.0

        total_longevity_cos_aux_anchor_loss = 0.0
        total_longevity_cos_aux_positive_loss = 0.0
        total_longevity_cos_aux_negative_loss = 0.0

        total_longevity_sin_aux_anchor_loss = 0.0
        total_longevity_sin_aux_positive_loss = 0.0
        total_longevity_sin_aux_negative_loss = 0.0

        num_batches = 0
        with torch.no_grad():
            for i, (anchor_piano, positive_piano, negative_piano,
                anchor_piano_quality_actual, positive_piano_quality_actual, negative_piano_quality_actual,
                anchor_piano_resonance_actual, positive_piano_resonance_actual, negative_piano_resonance_actual,
                anchor_piano_tension_actual, positive_piano_tension_actual, negative_piano_tension_actual,
                anchor_piano_longevity_cos_actual, positive_piano_longevity_cos_actual, negative_piano_longevity_cos_actual,
                anchor_piano_longevity_sin_actual, positive_piano_longevity_sin_actual, negative_piano_longevity_sin_actual,
                triplet_idx, anchor_piano_idx, positive_piano_idx, negative_piano_idx
            ) in enumerate(dataloader):
                
                if config.log_dataloader_triplet_idx:
                    self.log_triplet_records(
                        triplet_idx=triplet_idx,
                        anchor_idx=anchor_piano_idx,
                        positive_idx=positive_piano_idx,
                        negative_idx=negative_piano_idx,
                        logger=self.logger
                    )

                anchor_embedding, anchor_quality_logits, anchor_quality_recon, anchor_resonance_recon, anchor_tension_recon, anchor_longevity_cos_recon, anchor_longevity_sin_recon = model(*anchor_piano)
                positive_embedding, positive_quality_logits, positive_quality_recon, positive_resonance_recon, positive_tension_recon, positive_longevity_cos_recon, positive_longevity_sin_recon = model(*positive_piano)
                negative_embedding, negative_quality_logits, negative_quality_recon, negative_resonance_recon, negative_tension_recon, negative_longevity_cos_recon, negative_longevity_sin_recon = model(*negative_piano)

                #L2-normalize emb
                anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
                positive_embedding = F.normalize(positive_embedding, p=2, dim=1)
                negative_embedding = F.normalize(negative_embedding, p=2, dim=1)

                #aux losses
                quality_aux_anchor_loss = F.cross_entropy(anchor_quality_logits, anchor_piano_quality_actual)
                quality_aux_positive_loss = F.cross_entropy(positive_quality_logits, positive_piano_quality_actual)
                quality_aux_negative_loss = F.cross_entropy(negative_quality_logits, negative_piano_quality_actual)

                resonance_aux_anchor_loss = F.mse_loss(anchor_resonance_recon, anchor_piano_resonance_actual.unsqueeze(1))
                resonance_aux_positive_loss = F.mse_loss(positive_resonance_recon, positive_piano_resonance_actual.unsqueeze(1))
                resonance_aux_negative_loss = F.mse_loss(negative_resonance_recon, negative_piano_resonance_actual.unsqueeze(1))

                tension_aux_anchor_loss = F.mse_loss(anchor_tension_recon, anchor_piano_tension_actual.unsqueeze(1))
                tension_aux_positive_loss = F.mse_loss(positive_tension_recon, positive_piano_tension_actual.unsqueeze(1))
                tension_aux_negative_loss = F.mse_loss(negative_tension_recon, negative_piano_tension_actual.unsqueeze(1))

                longevity_cos_aux_anchor_loss = F.mse_loss(anchor_longevity_cos_recon, anchor_piano_longevity_cos_actual.unsqueeze(1))
                longevity_cos_aux_positive_loss = F.mse_loss(positive_longevity_cos_recon, positive_piano_longevity_cos_actual.unsqueeze(1))
                longevity_cos_aux_negative_loss = F.mse_loss(negative_longevity_cos_recon, negative_piano_longevity_cos_actual.unsqueeze(1))

                longevity_sin_aux_anchor_loss = F.mse_loss(anchor_longevity_sin_recon, anchor_piano_longevity_sin_actual.unsqueeze(1))
                longevity_sin_aux_positive_loss = F.mse_loss(positive_longevity_sin_recon, positive_piano_longevity_sin_actual.unsqueeze(1))
                longevity_sin_aux_negative_loss = F.mse_loss(negative_longevity_sin_recon, negative_piano_longevity_sin_actual.unsqueeze(1))

                #cum aux losses for later avg
                total_quality_aux_anchor_loss += quality_aux_anchor_loss.item()
                total_quality_aux_positive_loss += quality_aux_positive_loss.item()
                total_quality_aux_negative_loss += quality_aux_negative_loss.item()
                
                total_resonance_aux_anchor_loss += resonance_aux_anchor_loss.item()
                total_resonance_aux_positive_loss += resonance_aux_positive_loss.item()
                total_resonance_aux_negative_loss += resonance_aux_negative_loss.item()
                
                total_tension_aux_anchor_loss += tension_aux_anchor_loss.item()
                total_tension_aux_positive_loss += tension_aux_positive_loss.item()
                total_tension_aux_negative_loss += tension_aux_negative_loss.item()
                
                total_longevity_cos_aux_anchor_loss += longevity_cos_aux_anchor_loss.item()
                total_longevity_cos_aux_positive_loss += longevity_cos_aux_positive_loss.item()
                total_longevity_cos_aux_negative_loss += longevity_cos_aux_negative_loss.item()
                
                total_longevity_sin_aux_anchor_loss += longevity_sin_aux_anchor_loss.item()
                total_longevity_sin_aux_positive_loss += longevity_sin_aux_positive_loss.item()
                total_longevity_sin_aux_negative_loss += longevity_sin_aux_negative_loss.item()
                
                num_batches += 1

                #calc ecludian dist
                dist_anchor_pos = torch.norm(anchor_embedding - positive_embedding, p=2, dim=1)
                dist_anchor_neg = torch.norm(anchor_embedding - negative_embedding, p=2, dim=1)
                dist_pos_neg = torch.norm(positive_embedding - negative_embedding, p=2, dim=1)
                
                learned_representation_sim_anchor_pos = torch.clamp(1 - dist_anchor_pos / margin, min=0, max=1)
                learned_representation_sim_anchor_neg = torch.clamp(1 - dist_anchor_neg / margin, min=0, max=1)
                learned_representation_sim_pos_neg = torch.clamp(1 - dist_pos_neg / margin, min=0, max=1)
                
                dist_anchor_pos_list.extend(dist_anchor_pos.cpu().tolist())
                dist_anchor_neg_list.extend(dist_anchor_neg.cpu().tolist())
                dist_pos_neg_list.extend(dist_pos_neg.cpu().tolist())
                # print("learned_representation_sim_anchor_pos",learned_representation_sim_anchor_pos)
                # print("learned_representation_sim_anchor_neg",learned_representation_sim_anchor_neg)
                # print("learned_representation_sim_pos_neg",learned_representation_sim_pos_neg)
                
                #accumulate learned_representation similarities for final AUROC
                learned_representation_sim_anchor_pos_list.append(learned_representation_sim_anchor_pos)
                learned_representation_sim_anchor_neg_list.append(learned_representation_sim_anchor_neg)
                learned_representation_sim_pos_neg_list.append(learned_representation_sim_pos_neg)

                #Labels: all 1 cos its compared as < and > is applied vs threshold
                anchor_pos_labels = torch.ones_like(learned_representation_sim_anchor_pos)
                anchor_neg_labels = torch.ones_like(learned_representation_sim_anchor_neg)
                pos_neg_labels = torch.ones_like(learned_representation_sim_pos_neg)

                #update quality accuracy
                quality_accuracy_anchor.update(anchor_quality_logits.argmax(dim=1), anchor_piano_quality_actual)
                quality_accuracy_pos.update(positive_quality_logits.argmax(dim=1), positive_piano_quality_actual)
                quality_accuracy_neg.update(negative_quality_logits.argmax(dim=1), negative_piano_quality_actual)

                #threshold similarities for classification
                learned_representation_labels_anchor_pos = (learned_representation_sim_anchor_pos >= config.contrastive_threshold).float()
                learned_representation_labels_anchor_neg = (learned_representation_sim_anchor_neg < config.contrastive_threshold).float()
                learned_representation_labels_pos_neg = (learned_representation_sim_pos_neg < config.contrastive_threshold).float()

                #update precision/recall metrics
                precision_metric_anchor_pos.update(learned_representation_labels_anchor_pos, anchor_pos_labels)
                precision_metric_anchor_neg.update(learned_representation_labels_anchor_neg, anchor_neg_labels)
                precision_metric_pos_neg.update(learned_representation_labels_pos_neg, pos_neg_labels)
                
                recall_metric_anchor_pos.update(learned_representation_labels_anchor_pos, anchor_pos_labels)
                recall_metric_anchor_neg.update(learned_representation_labels_anchor_neg, anchor_neg_labels)
                recall_metric_pos_neg.update(learned_representation_labels_pos_neg, pos_neg_labels)

                #update mse metrics
                mse_metric_anchor_pos.update(learned_representation_sim_anchor_pos, torch.ones_like(learned_representation_sim_anchor_pos))
                mse_metric_anchor_neg.update(learned_representation_sim_anchor_neg, torch.zeros_like(learned_representation_sim_anchor_neg))
                mse_metric_pos_neg.update(learned_representation_sim_pos_neg, torch.zeros_like(learned_representation_sim_pos_neg))

                #classification check
                correct_anchor = (learned_representation_labels_anchor_pos == 1).float()
                correct_pos = (learned_representation_labels_anchor_pos == 1).float()
                correct_neg = (learned_representation_labels_anchor_neg == 1).float()

                total_correct_anchor += correct_anchor.sum().item()
                total_correct_pos += correct_pos.sum().item()
                total_correct_neg += correct_neg.sum().item()
                num_samples += learned_representation_labels_anchor_pos.size(0)

                correct_anchor_list.extend(correct_anchor.cpu().tolist())
                correct_pos_list.extend(correct_pos.cpu().tolist())
                correct_neg_list.extend(correct_neg.cpu().tolist())

                #df of batch stats
                batch_df = pd.DataFrame({
                    "raw_anchor_quality": anchor_piano_quality_actual.cpu().numpy().flatten(),
                    "anchor_quality_recon_class": torch.argmax(anchor_quality_logits, dim=1).cpu().numpy(),

                    "raw_positive_quality": positive_piano_quality_actual.cpu().numpy().flatten(),
                    "positive_quality_recon_class": torch.argmax(positive_quality_logits, dim=1).cpu().numpy(),

                    "raw_negative_quality": negative_piano_quality_actual.cpu().numpy().flatten(),
                    "negative_quality_recon_class": torch.argmax(negative_quality_logits, dim=1).cpu().numpy(),

                    "raw_anchor_resonance": anchor_piano_resonance_actual.cpu().numpy().flatten(),
                    "anchor_resonance_recon_class": anchor_resonance_recon.cpu().numpy().flatten(),

                    "raw_positive_resonance": positive_piano_resonance_actual.cpu().numpy().flatten(),
                    "positive_resonance_recon_class": positive_resonance_recon.cpu().numpy().flatten(),

                    "raw_negative_resonance": negative_piano_resonance_actual.cpu().numpy().flatten(),
                    "negative_resonance_recon_class": negative_resonance_recon.cpu().numpy().flatten(),

                    "raw_anchor_tension": anchor_piano_tension_actual.cpu().numpy().flatten(),
                    "anchor_tension_recon_class": anchor_tension_recon.cpu().numpy().flatten(),

                    "raw_positive_tension": positive_piano_tension_actual.cpu().numpy().flatten(),
                    "positive_tension_recon_class": positive_tension_recon.cpu().numpy().flatten(),

                    "raw_negative_tension": negative_piano_tension_actual.cpu().numpy().flatten(),
                    "negative_tension_recon_class": negative_tension_recon.cpu().numpy().flatten(),

                    "raw_anchor_longevity_cos": anchor_piano_longevity_cos_actual.cpu().numpy().flatten(),
                    "anchor_longevity_cos_recon_class": anchor_longevity_cos_recon.cpu().numpy().flatten(),

                    "raw_positive_longevity_cos": positive_piano_longevity_cos_actual.cpu().numpy().flatten(),
                    "positive_longevity_cos_recon_class": positive_longevity_cos_recon.cpu().numpy().flatten(),

                    "raw_negative_longevity_cos": negative_piano_longevity_cos_actual.cpu().numpy().flatten(),
                    "negative_longevity_cos_recon_class": negative_longevity_cos_recon.cpu().numpy().flatten(),

                    "raw_anchor_longevity_sin": anchor_piano_longevity_sin_actual.cpu().numpy().flatten(),
                    "anchor_longevity_sin_recon_class": anchor_longevity_sin_recon.cpu().numpy().flatten(),

                    "raw_positive_longevity_sin": positive_piano_longevity_sin_actual.cpu().numpy().flatten(),
                    "positive_longevity_sin_recon_class": positive_longevity_sin_recon.cpu().numpy().flatten(),

                    "raw_negative_longevity_sin": negative_piano_longevity_sin_actual.cpu().numpy().flatten(),
                    "negative_longevity_sin_recon_class": negative_longevity_sin_recon.cpu().numpy().flatten(),
                })

                all_features_reconstruction_dfs.append(batch_df)

        #final metric computations
        all_learned_representations_sim_anchor_pos = torch.cat(learned_representation_sim_anchor_pos_list, dim=0)
        all_learned_representations_sim_anchor_neg = torch.cat(learned_representation_sim_anchor_neg_list, dim=0)
        all_learned_representations_sim_pos_neg = torch.cat(learned_representation_sim_pos_neg_list, dim=0)

        # True labels for final metrics
        # Anchor-Pos => 1s, Anchor-Neg => 0s, Pos-Neg => 0s
        all_gtruth_labels_anchor_pos = torch.ones_like(all_learned_representations_sim_anchor_pos)
        all_gtruth_labels_anchor_neg = torch.zeros_like(all_learned_representations_sim_anchor_neg)
        all_gtruth_labels_pos_neg = torch.zeros_like(all_learned_representations_sim_pos_neg)
        
        #Combined AUROC
        #concatenate learned_representation similarities and labels 
        combined_learned_representations_sim = torch.cat([
            all_learned_representations_sim_anchor_pos,
            all_learned_representations_sim_anchor_neg,
            all_learned_representations_sim_pos_neg
        ], dim=0)
        
        combined_gtruth_labels = torch.cat([
            all_gtruth_labels_anchor_pos,
            all_gtruth_labels_anchor_neg,
            all_gtruth_labels_pos_neg
        ], dim=0)
        
        #calc combined AUROC
        combined_auroc_metric.update(combined_learned_representations_sim, combined_gtruth_labels)
        combined_auroc_score = combined_auroc_metric.compute()
        
        final_precision_anchor_pos = precision_metric_anchor_pos.compute()
        final_precision_anchor_neg = precision_metric_anchor_neg.compute()
        final_precision_pos_neg = precision_metric_pos_neg.compute()
        
        final_recall_anchor_pos = recall_metric_anchor_pos.compute()
        final_recall_anchor_neg = recall_metric_anchor_neg.compute()
        final_recall_pos_neg = recall_metric_pos_neg.compute()
        
        final_mse_anchor_pos = mse_metric_anchor_pos.compute()
        final_mse_anchor_neg = mse_metric_anchor_neg.compute()
        final_mse_pos_neg = mse_metric_pos_neg.compute()
        
        #results df
        results_df = pd.DataFrame()
        results_df["Ecludian_Distance_AP"] = dist_anchor_pos_list
        results_df["Ecludian_Distance_AN"] = dist_anchor_neg_list
        results_df["Ecludian_Distance_PN"] = dist_pos_neg_list
        
        results_df["Learned_Representation_Similarity_AP"] = all_learned_representations_sim_anchor_pos.cpu().tolist()
        results_df["Learned_Representation_Similarity_AN"] = all_learned_representations_sim_anchor_neg.cpu().tolist()
        results_df["Learned_Representation_Similarity_PN"] = all_learned_representations_sim_pos_neg.cpu().tolist()
        
        results_df["Correct_A"] = correct_anchor_list
        results_df["Correct_P"] = correct_pos_list
        results_df["Correct_N"] = correct_neg_list
        
        df_results_reset = results_df.reset_index(drop=True)
        df_results_reset.index = df_results_reset.index + 1

        self.plot_instance.plot_distance_distributions_for_epoch(
            epoch,
            dist_anchor_pos_list,
            dist_anchor_neg_list,
            "Eval"
        )
        
        #calc classification accuracies
        accuracy_anchor = (total_correct_anchor / num_samples) * 100 if num_samples > 0 else 0.0
        accuracy_pos = (total_correct_pos / num_samples) * 100 if num_samples > 0 else 0.0
        accuracy_neg = (total_correct_neg / num_samples) * 100 if num_samples > 0 else 0.0
        
        #avg aux losses
        avg_quality_aux_anchor_loss = total_quality_aux_anchor_loss / num_batches if num_batches else 0.0
        avg_quality_aux_positive_loss = total_quality_aux_positive_loss / num_batches if num_batches else 0.0
        avg_quality_aux_negative_loss = total_quality_aux_negative_loss / num_batches if num_batches else 0.0
        
        avg_resonance_aux_anchor_loss = total_resonance_aux_anchor_loss / num_batches if num_batches else 0.0
        avg_resonance_aux_positive_loss = total_resonance_aux_positive_loss / num_batches if num_batches else 0.0
        avg_resonance_aux_negative_loss = total_resonance_aux_negative_loss / num_batches if num_batches else 0.0
        
        avg_tension_aux_anchor_loss = total_tension_aux_anchor_loss / num_batches if num_batches else 0.0
        avg_tension_aux_positive_loss = total_tension_aux_positive_loss / num_batches if num_batches else 0.0
        avg_tension_aux_negative_loss = total_tension_aux_negative_loss / num_batches if num_batches else 0.0
        
        avg_longevity_cos_aux_anchor_loss = total_longevity_cos_aux_anchor_loss / num_batches if num_batches else 0.0
        avg_longevity_cos_aux_positive_loss = total_longevity_cos_aux_positive_loss / num_batches if num_batches else 0.0
        avg_longevity_cos_aux_negative_loss = total_longevity_cos_aux_negative_loss / num_batches if num_batches else 0.0
        
        avg_longevity_sin_aux_anchor_loss = total_longevity_sin_aux_anchor_loss / num_batches if num_batches else 0.0
        avg_longevity_sin_aux_positive_loss = total_longevity_sin_aux_positive_loss / num_batches if num_batches else 0.0
        avg_longevity_sin_aux_negative_loss = total_longevity_sin_aux_negative_loss / num_batches if num_batches else 0.0

        #move dist to tensor
        dist_anchor_pos_tensor = torch.tensor(dist_anchor_pos_list)
        dist_anchor_neg_tensor = torch.tensor(dist_anchor_neg_list)
        dist_pos_neg_tensor = torch.tensor(dist_pos_neg_list)

        if epoch % config.mid_training_operations_epoch_freq == 0:
            self.print_final_results(
                df_results_reset, accuracy_anchor, accuracy_pos, accuracy_neg,
                combined_auroc_metric, combined_auroc_score,
                final_precision_anchor_pos, final_precision_anchor_neg, final_precision_pos_neg,
                final_recall_anchor_pos, final_recall_anchor_neg, final_recall_pos_neg,
                final_mse_anchor_pos, final_mse_anchor_neg, final_mse_pos_neg,
                quality_accuracy_anchor, quality_accuracy_pos, quality_accuracy_neg,
                avg_resonance_aux_anchor_loss, avg_resonance_aux_positive_loss, avg_resonance_aux_negative_loss,
                avg_tension_aux_anchor_loss, avg_tension_aux_positive_loss, avg_tension_aux_negative_loss,
                avg_longevity_cos_aux_anchor_loss, avg_longevity_cos_aux_positive_loss, avg_longevity_cos_aux_negative_loss,
                avg_longevity_sin_aux_anchor_loss, avg_longevity_sin_aux_positive_loss, avg_longevity_sin_aux_negative_loss,
                dist_anchor_pos_tensor, dist_anchor_neg_tensor, dist_pos_neg_tensor,
                all_learned_representations_sim_anchor_pos, all_learned_representations_sim_anchor_neg, all_learned_representations_sim_pos_neg,
                avg_quality_aux_anchor_loss, avg_quality_aux_positive_loss, avg_quality_aux_negative_loss
            )

        self.logger.info("\n*** Evaluation Feature Reconstructions ***\n")
        all_features_reconstruction_df = pd.concat(all_features_reconstruction_dfs, ignore_index=True)
        
        #save features_reconstructions
        if epoch % config.mid_training_operations_epoch_freq == 0:
            csv_file_path = os.path.join(config.PLOTS_DIR, f"eval_reconstructions.csv")
            all_features_reconstruction_df.to_csv(csv_file_path, index=False)
            self.logger.info(f"Evaluation Feature Reconstructions saved to {csv_file_path}")

        #plot AUROC
        combined_learned_representation_labels = (combined_learned_representations_sim >= config.contrastive_threshold).long()
        combined_gtruth_labels = combined_gtruth_labels.cpu().numpy()
        combined_learned_representations_sim = combined_learned_representations_sim.cpu().numpy()
        combined_learned_representation_labels = combined_learned_representation_labels.cpu().numpy()

        if epoch % config.mid_training_operations_epoch_freq == 0:
            
            self.plot_instance.plot_auroc("tne_auroc", combined_gtruth_labels, combined_learned_representations_sim, epoch)
            self.plot_instance.plot_confusion_matrix("tne_cm",f"Confusion Matrix Epoch {epoch}",combined_gtruth_labels, combined_learned_representation_labels)

        self.metrics["eval_quality_xentropy_anchor"].append(avg_quality_aux_anchor_loss)
        self.metrics["eval_quality_xentropy_pos"].append(avg_quality_aux_positive_loss)
        self.metrics["eval_quality_xentropy_neg"].append(avg_quality_aux_negative_loss)
        self.metrics["eval_resonance_mse_anchor"].append(avg_resonance_aux_anchor_loss)
        self.metrics["eval_resonance_mse_pos"].append(avg_resonance_aux_positive_loss)
        self.metrics["eval_resonance_mse_neg"].append(avg_resonance_aux_negative_loss)
        self.metrics["eval_tension_mse_anchor"].append(avg_tension_aux_anchor_loss)
        self.metrics["eval_tension_mse_pos"].append(avg_tension_aux_positive_loss)
        self.metrics["eval_tension_mse_neg"].append(avg_tension_aux_negative_loss)
        self.metrics["eval_longevity_cos_mse_anchor"].append(avg_longevity_cos_aux_anchor_loss)
        self.metrics["eval_longevity_cos_mse_pos"].append(avg_longevity_cos_aux_positive_loss)
        self.metrics["eval_longevity_cos_mse_neg"].append(avg_longevity_cos_aux_negative_loss)
        self.metrics["eval_longevity_sin_mse_anchor"].append(avg_longevity_sin_aux_anchor_loss)
        self.metrics["eval_longevity_sin_mse_pos"].append(avg_longevity_sin_aux_positive_loss)
        self.metrics["eval_longevity_sin_mse_neg"].append(avg_longevity_sin_aux_negative_loss)

        if epoch % 10 == 0:
            self.plot_instance.plot_paired_metrics_bar(train_metrics, self.metrics)

        self.accuracy["eval_accuracy_anchor"].append(accuracy_anchor)
        self.accuracy["eval_accuracy_pos"].append(accuracy_pos)
        self.accuracy["eval_accuracy_neg"].append(accuracy_neg)

        if epoch % config.mid_training_operations_epoch_freq == 0:
            self.plot_instance.plot_paired_accuracy_bar(epoch, self.accuracy)

        #mem management
        dist_anchor_pos_list.clear()
        dist_anchor_neg_list.clear()
        dist_pos_neg_list.clear()
        scaled_sim_anchor_pos_list.clear()
        scaled_sim_anchor_neg_list.clear()
        scaled_sim_pos_neg_list.clear()
        learned_representation_sim_anchor_pos_list.clear()
        learned_representation_sim_anchor_neg_list.clear()
        learned_representation_sim_pos_neg_list.clear()
        correct_anchor_list.clear()
        correct_pos_list.clear()
        correct_neg_list.clear()
        all_features_reconstruction_dfs.clear()
        torch.cuda.empty_cache()

        self.logger.info("\n*** Evaluation Complete ***\n")
