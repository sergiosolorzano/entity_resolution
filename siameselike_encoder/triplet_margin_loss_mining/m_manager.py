import os, sys
import torch
import numpy as np
import pandas as pd
import config

np.random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from context import ER_Context
from utils.logger import Tne_Logger

if __name__ == "__main__":
    
    device = torch.device(f"cuda:{config.cuda_gpu}" if torch.cuda.is_available() else "cpu")
    print("GPU",torch.cuda.get_device_name(config.cuda_gpu))
    print("Device:", device)

    os.makedirs(os.path.join(config.PLOTS_DIR, "emb_frames"), exist_ok=True)

    ctx = ER_Context(device)
    ctx.logging_instance = Tne_Logger("tne_log")

    tne_train_paths = ctx.data_path_provider_instance.get_paths(mode='tne_mode', is_training=True)
    tne_eval_paths = ctx.data_path_provider_instance.get_paths(mode='tne_mode', is_training=False)

    if config.run_inference_hierarchical_clustering:
        ctx.logging_instance.info("\n***Run Hierarchical Clustering***")
        embeddings = ctx.inference_engine_instance.run_inference(ctx.data_path_provider_instance, config.inputs_config, tne_train_paths)
        embeddings = np.array(embeddings).squeeze(0)

        ctx.plots_instance.plot_distance_similarity_matrix(embeddings)

        ctx.hierarchical_clustering_instance.hierarchical_cluster_embeddings(embeddings=embeddings,
                                                                            n_clusters=None,
                                                                            distance_threshold=0.2,
                                                                            linkage='ward')
    else: #train and eval
        #create data stats
        ctx.logging_instance.info("***Normalized Data To Compare***")
        train_numeric_features_normd, train_labels_filtered = ctx.feature_engineering_instance.normalize_numeric_data(tne_train_paths["data_path"],
                                    config.inputs_config["numeric_features"], tne_train_paths["labels_path"], True) 
        
        eval_numeric_features_normd, eval_labels_filtered = ctx.feature_engineering_instance.normalize_numeric_data(tne_eval_paths["data_path"],
                                    config.inputs_config["numeric_features"], tne_eval_paths["labels_path"], False) 
        
        if not config.run_inference_hierarchical_clustering:
            ctx.data_metrics_instance.compare_data_original(tne_train_paths["data_path"], tne_train_paths["labels_path"],
                          tne_eval_paths["data_path"], tne_eval_paths["labels_path"], "original")
            ctx.data_metrics_instance.compare_data(train_numeric_features_normd, train_labels_filtered,
                        eval_numeric_features_normd, eval_labels_filtered, "transf")

        #train    
        if config.train_tabular_numeric_encoder:

            print("***Generate Train Dataloader***")
            train_dataloader = ctx.common_data_loader_instance.create_pairs_dataset_manager(
                config.inputs_config,
                tne_train_paths,
                True
            )
            
            ctx.plots_instance.plot_tension_histogram_from_dataloader(train_dataloader, "train")
            ctx.plots_instance.plot_resonance_histogram_from_dataloader(train_dataloader, "train")
            ctx.plots_instance.plot_quality_histogram_from_dataloader(train_dataloader, "train")
            ctx.plots_instance.plot_longevitycos_histogram_from_dataloader(train_dataloader, "train")
            ctx.plots_instance.plot_longevitysin_histogram_from_dataloader(train_dataloader, "train")

            print("***Generate Evaluation Dataloader***")
            eval_dataloader = ctx.common_data_loader_instance.create_pairs_dataset_manager(
                config.inputs_config,
                tne_eval_paths,
                False
            )
            
            ctx.plots_instance.plot_tension_histogram_from_dataloader(eval_dataloader, "eval")
            ctx.plots_instance.plot_resonance_histogram_from_dataloader(eval_dataloader, "eval")
            ctx.plots_instance.plot_quality_histogram_from_dataloader(eval_dataloader, "eval")
            ctx.plots_instance.plot_longevitycos_histogram_from_dataloader(eval_dataloader, "eval")
            ctx.plots_instance.plot_longevitysin_histogram_from_dataloader(eval_dataloader, "eval")

            #train numeric encoder
            print(f"\n***Train Tabular_Numeric_Encoder with {tne_train_paths["data_path"]} ***")
            ctx.training_instance.train_pairs_T_encoder(train_dataloader, eval_dataloader)