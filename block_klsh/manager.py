import pandas as pd
import numpy as np
import random

from context import ER_Context

from blocking.graph_conn import DeduP_Graph
from blocking.block_tree import BlockTree
from blocking.graph_visualization import visualize_graph_components

from clustering.klsh_perf_visualization import plot_elbow

import optimization.bayopt_weights as bayopt_weights

import config as config

np.random.seed(42)
random.seed(42)

if __name__ == "__main__":

    #instantiate classes
    ctx = ER_Context()
    klsh_engine_instance = ctx.klsh_engine_instance
    features_engineering_instance = ctx.feature_engineering_instance
    
    # Load and preprocess data
    data_df = pd.read_csv(config.source_data, encoding='cp1252')
    
    # Data type conversions
    data_df['name'] = data_df['name'].astype(str)
    data_df['tension'] = pd.to_numeric(data_df['tension'], errors='coerce')
    data_df['longevity'] = pd.to_datetime(data_df['longevity'], format="%d/%m/%Y", errors="coerce")
    print(data_df.head())
    
    #create tree and get strategy stages and specs
    blocker = BlockTree()
    blocking_rules_lib = blocker._load_blocking_rules(config.blocking_rules_fname)
    rule_stage_info_dict = blocker._register_scenario_blocking_rules(config.scenario_blocking_rules, blocking_rules_lib)
    
    #visualize pre-pruned graph
    blocker._fit_global_transformers(data_df)
    blocker._create_block_key(data_df)

    #visualize tree
    graph = DeduP_Graph()
    graph._create_graph(blocker.tree_blocks)
    graph._create_edges(blocker.tree_blocks)
    graph._visualize_graph_tree()

    #create preprune graph
    graph_weights = blocker._track_pair_provenance_and_weights()
    _, _, singleton_list = visualize_graph_components(graph_weights, f"{config.graph_dir}/preprun_components_graph", "Pre-prunning Provenance Components Graph", blocker.prepruned_graph_provenance)
    ctx.klsh_engine_instance.singleton_list = singleton_list

    #prune graph
    pruned_pairs = blocker._prune_graph()
    pruned_nx_graph, component_pairs_and_singletons_list, singleton_list = visualize_graph_components(blocker.pruned_graph, f"{config.graph_dir}/Pruned_components_graph", "Prunned Components Provenance Graph", blocker.prepruned_graph_provenance)
    ctx.klsh_engine_instance.singleton_list = singleton_list

    if config.verbose:
        print("Final Blocking Components Pairs", component_pairs_and_singletons_list)

    components_df = []
    for i in range(len(component_pairs_and_singletons_list)):
        df = pd.DataFrame(columns=data_df.columns)
        if len(component_pairs_and_singletons_list[i]) > 1:
            idx_set = set(idx for pair in component_pairs_and_singletons_list[i] for idx in pair)
            df = data_df.iloc[list(idx_set)]
        else:
            idx = component_pairs_and_singletons_list[i]
            df = data_df.iloc[list(idx)]
        components_df.append(df)

    #train LLSH
    if config.uniform_weights_klsh:

        df_numeric, pred_clusters, metrics_pairs_result_df, lowest_k_highest_F1_row = klsh_engine_instance.predict_klsh_clusters_and_metrics(component_pairs_and_singletons_list, data_df, config.component_idx, config.k_bottom, config.k_top)
        
        print("\n== Final Results ==")
        if lowest_k_highest_F1_row is not None:
            print("\nFeature_weights:\n", config.feature_weights)

            print("\nPerformance Metric Pair Results:\n", metrics_pairs_result_df)
            print("\nLowest K Highest F-1:\n", lowest_k_highest_F1_row)

            if len(klsh_engine_instance.wcss_dict) > 0:
                klsh_engine_instance.best_elbow_k = ctx.perf_metrics_instance.find_best_elbok_k(klsh_engine_instance.wcss_dict)
                print("Knee WCSS", klsh_engine_instance.wcss_dict)
                print("Knee best elbow", klsh_engine_instance.best_elbow_k)
                plot_elbow(klsh_engine_instance.wcss_dict, config.component_idx)
            print("Best silhouette Score:",klsh_engine_instance.best_sil_score, "Best silhouette K:", klsh_engine_instance.best_sil_k)

            print("\n== Final Clusters: ==")
            for k, k_cluster_records in ctx.klsh_engine_instance.final_klsh_clusters.items():
                print(f"k={k}: ",k_cluster_records)
            print("Singletons:",ctx.klsh_engine_instance.singleton_list)
            
            print("=== End of Training ===")
        else:
            print(f"No Precision Stats to report for component {config.component_idx} holding records {component_pairs_and_singletons_list[config.component_idx]}")

        exit(0)
    
    #bayesian opt for weights
    if config.uniform_weights_klsh==False and config.bayesian_optimization_klsh==True:
        
        bayopt = bayopt_weights.BayOpt(component_pairs_and_singletons_list, 
                                       data_df, config.k_bottom, config.k_top)
        best_weights = bayopt.run_klsh_bayopt(ctx)
        print("\nFinal Best (avg) Bayesian Weights:",best_weights)

        print("=== End of Bayesian Optimization ===")
        exit(0)
        
    if config.uniform_weights_klsh==False and config.bayesian_optimization_klsh==False and config.test_optimized_weights_klsh==True:
        
        #hard code bayesian opt weights
        config.feature_weights = {
                'tension_adj_cos': config.best_weights[0],
                'tension_adj_sin': config.best_weights[1],
                'tension': config.best_weights[2],
                'resonance': config.best_weights[3],
                'longevity_cos': config.best_weights[4],
                'longevity_sin': config.best_weights[5],
                'quality_cos': config.best_weights[6],
                'quality_sin': config.best_weights[7],
                'amt_sold': config.best_weights[8]
            }
        
        df_numeric, pred_clusters, metrics_pairs_result_df, lowest_k_highest_F1_row = klsh_engine_instance.predict_klsh_clusters_and_metrics(component_pairs_and_singletons_list, data_df, config.component_idx, config.k_bottom, config.k_top)
        
        print("\n== Final Results ==")
        print("\nFeature_weights:\n", config.feature_weights)

        print("\nPerformance Metric Pair Results:\n",metrics_pairs_result_df)
        print("\nLowest K Highest F-1:\n", lowest_k_highest_F1_row)
        
        if len(klsh_engine_instance.wcss_dict) > 0:
            klsh_engine_instance.best_elbow_k = ctx.perf_metrics_instance.find_best_elbok_k(klsh_engine_instance.wcss_dict)
            print("Knee WCSS", klsh_engine_instance.wcss_dict)
            print("Knee best elbow", klsh_engine_instance.best_elbow_k)
            plot_elbow(klsh_engine_instance.wcss_dict, config.component_idx)
        
        plot_elbow(klsh_engine_instance.wcss_dict, config.component_idx)
        print("Best silhouette Score:",klsh_engine_instance.best_sil_score, "Best silhouette K:", klsh_engine_instance.best_sil_k)
        
        print("\n== Final Clusters: ==")
        for k, k_cluster_records in ctx.klsh_engine_instance.final_klsh_clusters.items():
            print(f"k={k}: ",k_cluster_records)
        print("Singletons:",ctx.klsh_engine_instance.singleton_list)
        
        print("=== End of Evaluation ===")
        exit(0)

        


