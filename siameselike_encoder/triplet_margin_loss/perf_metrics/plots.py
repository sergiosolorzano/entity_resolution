import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import umap.umap_ as umap
import subprocess

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg') 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
import config

class Plots:
    def __init__(self, logger, device):
        self.device = device
        self.logger = logger

    def plot_auroc(self, fname, ground_truth_labels, learned_representation_similarities, epoch):
        
        colour = "darkorange"
        
        unique_labels = np.unique(ground_truth_labels)
        if len(unique_labels) < 2:
            self.logger.info(f"[Error] need 2 classes (0 and 1) for `true_labels`, found classes: {unique_labels}")
            return
        
        #calc roc, auc
        fpr, tpr, thresholds = roc_curve(ground_truth_labels, learned_representation_similarities)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=colour, lw=2, label=f"AUROC (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FP/(FP+TN))')
        plt.ylabel('True Positive Rate (TP/(FP+FN))')
        plt.title(f'TNE Eval Receiver Operating Characteristic (ROC) Epoch {epoch}')
        plt.legend(loc="lower right")
        plt.grid()

        #note thresholds
        if len(thresholds) > 1:
            step = max(1, len(thresholds) // 10)
            for i in range(len(thresholds)):
                if i % step == 0 or i == len(thresholds) - 1:
                    plt.text(fpr[i], tpr[i], f"{thresholds[i]:.2f}", fontsize=8, ha='right', va='bottom')

        fname_path = os.path.join(config.PLOTS_DIR, fname + ".png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_confusion_matrix(self, fname, title, true_labels, learned_representation_labels):
        
        cm = confusion_matrix(true_labels, learned_representation_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Label 0", "Label 1"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        
        fname_path = os.path.join(config.PLOTS_DIR, fname+".png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_quality_histogram_from_dataloader(self, dataloader, data_type):
        
        anchor_qualities = []
        pos_qualities = []
        neg_qualities = []

        for batch in dataloader:
            _, _, _, anchor_quality, pos_quality, neg_quality, *rest = batch
            anchor_qualities.extend(anchor_quality.cpu().numpy())
            pos_qualities.extend(pos_quality.cpu().numpy())
            neg_qualities.extend(neg_quality.cpu().numpy())

        #bins for histogram
        bins = np.linspace(
            min(anchor_qualities + pos_qualities + neg_qualities),
            max(anchor_qualities + pos_qualities + neg_qualities),
            50
        )

        anchor_counts, bin_edges = np.histogram(anchor_qualities, bins=bins, density=True)
        pos_counts, _ = np.histogram(pos_qualities, bins=bins, density=True)
        neg_counts, _ = np.histogram(neg_qualities, bins=bins, density=True)

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, anchor_counts, label="Anchor quality", color="blue", linewidth=2)
        plt.plot(bin_centers, pos_counts, label="Positive quality", color="green", linestyle="--", linewidth=2)
        plt.plot(bin_centers, neg_counts, label="Negative quality", color="red", linestyle="-.", linewidth=2)

        plt.title(f"Quality Distribution from {data_type}_DataLoader", fontsize=14)
        plt.xlabel("Quality Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{data_type}_quality_hg.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_longevitycos_histogram_from_dataloader(self, dataloader, data_type):
        
        anchor_longevitiescos = []
        pos_longevitiescos = []
        neg_longevitiescos = []

        for batch in dataloader:
            _, _, _, _, _, _, _, _, _, _, _, _, anchor_longevitycos, pos_longevitycos, neg_longevitycos, *rest = batch

            anchor_longevitiescos.extend(anchor_longevitycos.cpu().numpy())
            pos_longevitiescos.extend(pos_longevitycos.cpu().numpy())
            neg_longevitiescos.extend(neg_longevitycos.cpu().numpy())

        #define bins histogram
        bins = np.linspace(
            min(anchor_longevitiescos + pos_longevitiescos + neg_longevitiescos),
            max(anchor_longevitiescos + pos_longevitiescos + neg_longevitiescos),
            50
        )

        anchor_counts, bin_edges = np.histogram(anchor_longevitiescos, bins=bins, density=True)
        pos_counts, _ = np.histogram(pos_longevitiescos, bins=bins, density=True)
        neg_counts, _ = np.histogram(neg_longevitiescos, bins=bins, density=True)

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, anchor_counts, label="Anchor longevity_cos", color="blue", linewidth=2)
        plt.plot(bin_centers, pos_counts, label="Positive longevity_cos", color="green", linestyle="--", linewidth=2)
        plt.plot(bin_centers, neg_counts, label="Negative longevity_cos", color="red", linestyle="-.", linewidth=2)

        plt.title(f"Longevity_Cos Distribution from {data_type}_DataLoader", fontsize=14)
        plt.xlabel("Longevity_Cos Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{data_type}_longevitycos_hg.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_longevitysin_histogram_from_dataloader(self, dataloader, data_type):
        
        anchor_longevitiessin = []
        pos_longevitiessin = []
        neg_longevitiessin = []

        for batch in dataloader:
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, anchor_longevitysin, pos_longevitysin, neg_longevitysin, *rest = batch

            anchor_longevitiessin.extend(anchor_longevitysin.cpu().numpy())
            pos_longevitiessin.extend(pos_longevitysin.cpu().numpy())
            neg_longevitiessin.extend(neg_longevitysin.cpu().numpy())

        #bins for histogram
        bins = np.linspace(
            min(anchor_longevitiessin + pos_longevitiessin + neg_longevitiessin),
            max(anchor_longevitiessin + pos_longevitiessin + neg_longevitiessin),
            50
        )

        anchor_counts, bin_edges = np.histogram(anchor_longevitiessin, bins=bins, density=True)
        pos_counts, _ = np.histogram(pos_longevitiessin, bins=bins, density=True)
        neg_counts, _ = np.histogram(neg_longevitiessin, bins=bins, density=True)

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, anchor_counts, label="Anchor longevity_sin", color="blue", linewidth=2)
        plt.plot(bin_centers, pos_counts, label="Positive longevity_sin", color="green", linestyle="--", linewidth=2)
        plt.plot(bin_centers, neg_counts, label="Negative longevity_sin", color="red", linestyle="-.", linewidth=2)

        plt.title(f"Longevity_Sin Distribution from {data_type}_DataLoader", fontsize=14)
        plt.xlabel("Longevity_Sin Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{data_type}_longevitysin_hg.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_tension_histogram_from_dataloader(self, dataloader, data_type):
        
        anchor_tensions = []
        pos_tensions = []
        neg_tensions = []

        for batch in dataloader:
            _, _, _, _, _, _, _, _, _, anchor_tension, pos_tension, neg_tension, *rest = batch

            anchor_tensions.extend(anchor_tension.cpu().numpy())
            pos_tensions.extend(pos_tension.cpu().numpy())
            neg_tensions.extend(neg_tension.cpu().numpy())

        #bins for histogram
        bins = np.linspace(
            min(anchor_tensions + pos_tensions + neg_tensions),
            max(anchor_tensions + pos_tensions + neg_tensions),
            50
        )

        anchor_counts, bin_edges = np.histogram(anchor_tensions, bins=bins, density=True)
        pos_counts, _ = np.histogram(pos_tensions, bins=bins, density=True)
        neg_counts, _ = np.histogram(neg_tensions, bins=bins, density=True)

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, anchor_counts, label="Anchor tensions", color="blue", linewidth=2)
        plt.plot(bin_centers, pos_counts, label="Positive tensions", color="green", linestyle="--", linewidth=2)
        plt.plot(bin_centers, neg_counts, label="Negative tensions", color="red", linestyle="-.", linewidth=2)

        plt.title(f"tension Distribution from {data_type}_DataLoader", fontsize=14)
        plt.xlabel("tension Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{data_type}_tension_hg.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_resonance_histogram_from_dataloader(self, dataloader, data_type):
        
        anchor_resonances = []
        pos_resonances = []
        neg_resonances = []

        for batch in dataloader:
            _, _, _, _, _, _, anchor_resonance, pos_resonance, neg_resonance, *rest = batch

            anchor_resonances.extend(anchor_resonance.cpu().numpy())
            pos_resonances.extend(pos_resonance.cpu().numpy())
            neg_resonances.extend(neg_resonance.cpu().numpy())

        #bins for histogram
        bins = np.linspace(
            min(anchor_resonances + pos_resonances + neg_resonances),
            max(anchor_resonances + pos_resonances + neg_resonances),
            50
        )

        anchor_counts, bin_edges = np.histogram(anchor_resonances, bins=bins, density=True)
        pos_counts, _ = np.histogram(pos_resonances, bins=bins, density=True)
        neg_counts, _ = np.histogram(neg_resonances, bins=bins, density=True)

        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, anchor_counts, label="Anchor resonance", color="blue", linewidth=2)
        plt.plot(bin_centers, pos_counts, label="Positive resonance", color="green", linestyle="--", linewidth=2)
        plt.plot(bin_centers, neg_counts, label="Negative resonance", color="red", linestyle="-.", linewidth=2)

        plt.title(f"resonance Distribution from {data_type}_DataLoader", fontsize=14)
        plt.xlabel("resonance Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{data_type}_resonance_hg.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_paired_metrics_bar(self, train_metrics, eval_metrics):
        metric_pairs = [
            ("train_quality_xentropy_anchor", "eval_quality_xentropy_anchor"),
            ("train_quality_xentropy_pos", "eval_quality_xentropy_pos"),
            ("train_quality_xentropy_neg", "eval_quality_xentropy_neg"),
            ("train_resonance_mse_anchor", "eval_resonance_mse_anchor"),
            ("train_resonance_mse_pos", "eval_resonance_mse_pos"),
            ("train_resonance_mse_neg", "eval_resonance_mse_neg"),
            ("train_tension_mse_anchor", "eval_tension_mse_anchor"),
            ("train_tension_mse_pos", "eval_tension_mse_pos"),
            ("train_tension_mse_neg", "eval_tension_mse_neg"),
            ("train_longevity_cos_mse_anchor", "eval_longevity_cos_mse_anchor"),
            ("train_longevity_cos_mse_pos", "eval_longevity_cos_mse_pos"),
            ("train_longevity_cos_mse_neg", "eval_longevity_cos_mse_neg"),
            ("train_longevity_sin_mse_anchor", "eval_longevity_sin_mse_anchor"),
            ("train_longevity_sin_mse_pos", "eval_longevity_sin_mse_pos"),
            ("train_longevity_sin_mse_neg", "eval_longevity_sin_mse_neg"),
        ]

        metrics = {**train_metrics, **eval_metrics}

        current_epoch = train_metrics["epoch"][-1]

        plt.figure(figsize=(16, 10))

        metric_names = []
        train_values = []
        eval_values = []

        #use last train/eval values
        for train_key, eval_key in metric_pairs:
            metric_names.append(train_key.replace("train_", ""))
            train_values.append(metrics[train_key][-1])
            eval_values.append(metrics[eval_key][-1])

        x = range(len(metric_names))

        bar_width = 0.4
        colour = "blue"
        plt.bar([i - bar_width / 2 for i in x], train_values, bar_width, label="Train", color=colour)
        plt.bar([i + bar_width / 2 for i in x], eval_values, bar_width, label="Eval", color="darkorange")

        plt.xticks(x, metric_names, rotation=45, ha="right", fontsize=12)
        plt.ylabel("MSE", fontsize=14)
        plt.title(f"Training vs Evaluation Metrics (Epoch {current_epoch})", fontsize=16)
        plt.legend(loc="upper left")

        plt.subplots_adjust(bottom=0.25)

        fname_path = os.path.join(config.PLOTS_DIR, "runtime_metrics_bar.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_paired_accuracy_bar(self, epoch, eval_metrics):
        metric_keys = [
            "eval_accuracy_anchor",
            "eval_accuracy_pos",
            "eval_accuracy_neg",
        ]

        plt.figure(figsize=(10, 6))

        metric_names = []
        eval_values = []

        for key in metric_keys:
            metric_names.append(key.replace("eval_", "").replace("_", " ").capitalize())
            eval_values.append(eval_metrics[key][-1])

        x = range(len(metric_names))

        colour = "blue"
        plt.bar(x, eval_values, color=colour, label="Eval")

        plt.xticks(x, metric_names, rotation=45, ha="right", fontsize=12)
        plt.ylabel("Accuracy (Count)", fontsize=14)
        plt.title(f"Evaluation Accuracy Metrics (Epoch {epoch})", fontsize=16)
        
        plt.ylim(0, 100)
        plt.yticks(range(0, 101, 10))  

        plt.legend(loc="upper left")

        plt.subplots_adjust(bottom=0.3)

        fname_path = os.path.join(config.PLOTS_DIR, "runtime_accuracy_bar.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_mining_triplet_survived(self, trip_percent_values):
        
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(trip_percent_values)))
        plt.plot(epochs, trip_percent_values, marker='o', linestyle='-', label='Survived Percent')

        plt.xlabel('Epochs')
        plt.ylabel('Surviving Mining Triplet Percent (%)')
        plt.title('Surviving Mining Triplet Evolution Over Epochs')
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.legend()
        plt.grid(True)
        fname_path = os.path.join(config.PLOTS_DIR, "surviving_mining_triplet.png")
        plt.savefig(fname_path)
        plt.clf()
        plt.close()

    def plot_distance_distributions_for_epoch(
        self,
        epoch,
        ap_distances,
        an_distances,
        simulation_type
        ):
    
        if torch.is_tensor(ap_distances):
            ap_distances = ap_distances.detach().cpu().numpy()
        if torch.is_tensor(an_distances):
            an_distances = an_distances.detach().cpu().numpy()
        
        ap_distances = np.array(ap_distances)
        an_distances = np.array(an_distances)

        plt.figure(figsize=(10, 6))
        
        #range to scale
        min_dist = min(ap_distances.min(), an_distances.min())
        max_dist = max(ap_distances.max(), an_distances.max())
        
        #histogram bins
        bins = np.linspace(min_dist, max_dist, 50)
        
        plt.hist(ap_distances, bins=bins, alpha=0.5, color='cornflowerblue', label='AP Distances', density=True)
        plt.hist(an_distances, bins=bins, alpha=0.3, color='#ffaaaa', label='AN Distances', density=True, edgecolor='red')#, hatch='///'

        plt.title(f"{simulation_type} Distance Distributions (Epoch {epoch})")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        fig_path = os.path.join(config.PLOTS_DIR, f"distance_distrib_{simulation_type}_e_{epoch}.png")
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()
        self.logger.info(f"{simulation_type} Distance distribution plot saved to {fig_path} for epoch {epoch}")

    def plot_training_loss(self, loss_history, similarity_loss_history, aux_loss_history, epoch):
        
        plt.figure(figsize=(10, 5))
        epochs = range(0, len(loss_history))
        plt.plot(epochs, loss_history, marker='o', linestyle='-', markersize=2, color='blue', label='Total Loss')
        plt.plot(epochs, similarity_loss_history, marker='s', linestyle='-', markersize=2, color='green', label='Similarity Loss')
        plt.plot(epochs, aux_loss_history, marker='^', linestyle='-', markersize=2, color='red', label='Aux Loss')
        
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0, 1.5)
        plt.title("Training Loss Over Epochs")
        plt.legend(loc='upper right')
        tick_positions = range(0, len(loss_history), 10)
        plt.xticks(tick_positions, rotation=45, fontsize=8, ha='right')
        plt.grid(True)

        fig_path = os.path.join(config.PLOTS_DIR, f"train_loss_hist.png")
        plt.savefig(fig_path)
        plt.clf()
        plt.close()

    def plot_combined_statistics(self, train_statistics, eval_statistics, group_name, title):
            
            features = train_statistics.index
            metrics = ["mean", "median", "std", "outlier_percentage"]

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                train_values = train_statistics[metric]
                eval_values = eval_statistics[metric]

                bar_width = 0.35
                x = range(len(features))

                axes[idx].bar(x, train_values, width=bar_width, label='Train', color='skyblue')
                axes[idx].bar([p + bar_width for p in x], eval_values, width=bar_width, label='Eval', color='orange')

                axes[idx].set_title(f"{metric.capitalize()} for {group_name.capitalize()}")
                axes[idx].set_ylabel(metric.capitalize())
                axes[idx].set_xticks([p + bar_width / 2 for p in x])
                axes[idx].set_xticklabels(features, rotation=45)
                axes[idx].legend()

            plt.suptitle(f"Statistics for {group_name.capitalize()}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            fname_path = os.path.join(config.DATA_PLOTS_DIR, f"{title}_{group_name.lower()}_stats.png")
            plt.savefig(fname_path)
            plt.clf()
            plt.close()

    def save_encoder_architecture(self, encoder_arch_str)    :

        encoder_arch_path = os.path.join(config.PLOTS_DIR, f"encoder_arch.txt")
        with open(encoder_arch_path, "w", encoding="utf-8") as f:
            f.write(encoder_arch_str)

    def plot_distance_similarity_matrix(self, embeddings):
        
        distance_matrix = np.zeros((len(embeddings), len(embeddings)))
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        for i in range(len(embeddings_tensor)):
            for j in range(i, len(embeddings_tensor)):
                dist = torch.norm(embeddings_tensor[i] - embeddings_tensor[j], p=2)
                distance_matrix[i][j]=dist.item()
                similarity_matrix[i][j] = torch.clamp(1 - (dist / config.contrastive_margin), min=0, max=1)
                #mirror
                if i!=j:
                    distance_matrix[j][i] = distance_matrix[i][j]
                    similarity_matrix[j][i] = similarity_matrix[i][j]

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap='viridis_r',
                        xticklabels=range(len(embeddings_tensor)), yticklabels=range(len(embeddings_tensor)))
        
        plt.title('Heatmap of Distance Matrix')
        plt.ylabel('record_id')
        plt.xlabel('record_id')
        
        # Save the figure
        plt.savefig(os.path.join(config.PLOTS_DIR, "distance_matrix_heatmap.png"), dpi=300)
        plt.clf()
        plt.close()

    def visualize_embedding_in_manifold(self, anchor_embedding, positive_embedding, negative_embedding, epoch):

        anchor = anchor_embedding.detach().cpu().numpy()
        positive = positive_embedding.detach().cpu().numpy()
        negative = negative_embedding.detach().cpu().numpy()

        embeddings = np.vstack([anchor, positive, negative])
        labels = (['anchor'] * len(anchor) +
                ['positive'] * len(positive) +
                ['negative'] * len(negative))

        #reduce dim
        reducer = umap.UMAP(
            # random_state=42, #rem for parallelism
            n_neighbors=15,
            min_dist=0.1,
            n_components=2, #2d proj
            metric='euclidean'
        )

        embeddings_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        for label in ['anchor', 'positive', 'negative']:
            idx = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, alpha=0.6)

        plt.title(f"Learned Representations Training visualized in a 2D Manifold - Epoch {epoch}")
        plt.legend()
        plt.tight_layout()

        frame_number = epoch // 1
        plt.savefig(os.path.join(config.PLOTS_DIR, "emb_frames", f"embed_dim_reduced_umap_epoch_{frame_number:04d}.png"), dpi=300)
        
        plt.clf()
        plt.close()

    def create_embedding_animation(self):
        ffmpeg_path = config.embed_mp4
        ffmpeg_cmd = [
            ffmpeg_path,
            '-y',
            '-framerate', '2',
            '-i', os.path.join(config.PLOTS_DIR, 'emb_frames', 'embed_dim_reduced_umap_epoch_%04d.png'),
            '-vf', 'fade=t=in:st=0:d=0.5:alpha=1,fade=t=out:st=4.5:d=0.5:alpha=1,format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            os.path.join(config.PLOTS_DIR, "embeddings_umap_evolution.mp4")
        ]
        subprocess.run(ffmpeg_cmd)
