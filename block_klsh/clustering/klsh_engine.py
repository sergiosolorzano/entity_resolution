import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx

from features.features_engineering import Feature_Engineering
from clustering.perf_metrics import Performance_Metrics
from blocking.graph_visualization import visualize_nx_graph

import config

class KLSH_Engine():
    def __init__(self, feature_engineering_instance: Feature_Engineering, performance_metrics_instance: Performance_Metrics):
        self.best_sil_score = -1
        self.best_sil_k = -1
        self.feature_engineering_instance = feature_engineering_instance
        self.perf_metrics_instance = performance_metrics_instance
        self.wcss_dict = {}
        self.best_elbow_k = -1
        self.singleton_list = []
        self.final_klsh_clusters = {}

    def predict_klsh_clusters_and_metrics(self, component_pairs_and_singletons_list, data_df: pd.DataFrame, 
                                          component_idx, k_bottom, k_top):
        
        self.final_klsh_clusters = {}

        #true values
        if component_idx==0: true_pairs = config.true_pairs_component_0
        elif component_idx==1: true_pairs = config.true_pairs_component_1
        elif component_idx==2: true_pairs = config.true_pairs_component_2
        # elif component_idx==3: true_pairs = config.true_pairs_component_3
        # elif component_idx==4: true_pairs = config.true_pairs_component_3
        else:
            print("Ground Truth Component Values not found, exiting.")
            exit(0)
        
        components_df = []
        for i in range(len(component_pairs_and_singletons_list)):
            df = pd.DataFrame(columns=data_df.columns)
            if len(component_pairs_and_singletons_list[i]) > 1:
                idx_set = set(idx for pair in component_pairs_and_singletons_list[i] for idx in pair)
            else:
                idx_set = component_pairs_and_singletons_list[i]
            df = data_df.iloc[list(idx_set)]
            components_df.append(df)

        #klsh
        print("\n=== KLSH Cluster Analysis ===")
        print("\nComponents pairs",component_pairs_and_singletons_list[component_idx])
        unique_component_records = np.unique(component_pairs_and_singletons_list[component_idx])
        print("unique_component_records",unique_component_records)

        k = k_bottom
        target_num_k = min(k_top, len(unique_component_records))+1 - k_bottom
        df_numeric = lowest_k_highest_F1_metrics = None
        metrics_pairs_result = []
        
        if len(unique_component_records) > 1:#pairs

            for k in range(k_bottom, min(k_top, len(unique_component_records))+1):

                if config.verbose:
                    print("\nK=",k)

                component_indices = set(np.array(component_pairs_and_singletons_list[component_idx]).flatten())

                component_df = data_df.iloc[list(component_indices)]
                df_numeric, pred_clusters, pred_cluster_records, pred_cluster_pairs, pred_pairs, similarity_matrix = self.klsh_embedding_numeric(component_df, k, target_num_k, config.feature_weights, self.feature_engineering_instance)
            
                if config.verbose:
                    print("pred_clusters",pred_clusters)
                    print("pred_cluster_records",pred_cluster_records)
                    print("pred_cluster_record_pairs",pred_cluster_pairs)
                    print("pred_pairs",pred_pairs)

                self.final_klsh_clusters[k] = pred_cluster_records

                #eval lsh
                self.graph_cluster(unique_component_records, pred_pairs, component_idx, k)
                metrics_pairs_result = self.perf_metrics_instance.calculate_metrics(pred_pairs, true_pairs, k, metrics_pairs_result)
                
        else: #singleton
            pred_clusters = 0
            pred_pairs = []
            pred_cluster_pairs = {0:None}
            pred_cluster_records = unique_component_records[0]
                
            if config.verbose:
                print("pred_clusters",pred_clusters)
                print("pred_cluster_records",pred_cluster_records)
                print("pred_cluster_record_pairs",pred_cluster_pairs)
                print("pred_pairs",pred_pairs)

            self.final_klsh_clusters[k] = pred_cluster_records

            #eval lsh
            self.graph_cluster(unique_component_records, pred_pairs, component_idx, k)
            metrics_pairs_result = self.perf_metrics_instance.calculate_metrics(pred_pairs, true_pairs, k, metrics_pairs_result)

        metrics_pairs_result_df = pd.DataFrame(metrics_pairs_result)
        
        lowest_k_highest_F1_metrics = metrics_pairs_result_df.loc[metrics_pairs_result_df["F1"].idxmax()]
        
        if config.verbose:
            print("metrics_pairs_result_df", metrics_pairs_result_df)
            print("lowest_k_highest_F1_metrics:\n",lowest_k_highest_F1_metrics)

        return df_numeric, pred_clusters, metrics_pairs_result_df, lowest_k_highest_F1_metrics
    
    def graph_cluster(self, unique_component_records, pred_pairs, component_idx, k):
        
        G = nx.Graph()

        #add all records in component to capture singletons
        G.add_nodes_from(set(unique_component_records))
        if len(pred_pairs) > 0:
            G.add_edges_from(pred_pairs)
        entities = list(nx.connected_components(G))
        if config.verbose:
            print("Component entities", entities)
        
        visualize_nx_graph(G, entities, f"{config.graph_dir}/klsh_entities_graph_k_{k}_comp_{component_idx}", f"KLSH Entities Graph K={k} Component {component_idx}")

    def create_records_similarity_matrix(self, df_numeric:pd.DataFrame):

        num_feature_array = np.array(df_numeric)

        cossim_matrix = cosine_similarity(num_feature_array)
        
        record_original_idx_list = df_numeric.index.tolist()

        cossim_matrix_df = pd.DataFrame(
            cossim_matrix,
            index=record_original_idx_list,
            columns=record_original_idx_list
        )

        if config.verbose:
            print("cossim_matrix_df\n", cossim_matrix_df)

        return cossim_matrix_df
    
    def klsh_embedding_numeric(self, data_df: pd.DataFrame, k, target_num_k, weights, feature_engineering_instance:Feature_Engineering, random_state=42):
        
        #transform features
        if config.verbose:
            print("datadf", data_df)
        
        df_numeric = data_df.copy()

        df_numeric = df_numeric.drop('name', axis=1)

        #tension, resonance, amt_sold fit to standardscaler
        for col in df_numeric.columns:
            if col != 'name' and col!='longevity' and col!='quality' and col!='tension_adj':
                df_numeric[[col]] = StandardScaler().fit_transform(df_numeric[[col]])
        
        #project bool
        cos_val, sin_val = self.feature_engineering_instance.embed_bool_category(df_numeric['tension_adj'])
        df_numeric['tension_adj_cos'] = cos_val * weights['tension_adj_cos']
        df_numeric['tension_adj_sin'] = sin_val * weights['tension_adj_sin']
        df_numeric = df_numeric.drop('tension_adj', axis=1)
        
        df_numeric['tension'] = df_numeric['tension'].values * weights['tension']
        df_numeric['resonance'] = df_numeric['resonance'].values * weights['resonance']

        cos_dt, sin_dt = feature_engineering_instance.quarter_circle_dt_projection_series(df_numeric['longevity'])
        df_numeric['longevity_cos'] = cos_dt * weights['longevity_cos']
        df_numeric['longevity_sin'] = sin_dt * weights['longevity_sin']
        df_numeric = df_numeric.drop('longevity', axis=1)

        #project ordinal categ
        #df_numeric['quality'] = df_numeric['quality'].values * weights['quality']
        cos_quality, sin_quality = self.feature_engineering_instance.embed_ordinal_category(df_numeric['quality'])
        df_numeric['quality_cos'] = cos_quality.values * weights['quality_cos']
        df_numeric['quality_sin'] = sin_quality.values * weights['quality_sin']
        df_numeric = df_numeric.drop('quality', axis=1)

        df_numeric['amt_sold'] = df_numeric['amt_sold'].values * weights['amt_sold']
        
        # Concatenate features
        num_features_array = np.array(df_numeric)

        # Calculate pairwise distances
        similarity_matrix = cosine_similarity(num_features_array)
        
        #TODO move so doesn't create each time
        all_records_cossim_matrix_df = self.create_records_similarity_matrix(df_numeric)
        #print("similarity_matrix",similarity_matrix)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        pred_clusters = kmeans.fit_predict(num_features_array)

        #Get record idx in df_numeric of cluster result
        cluster_records = {}
        cluster_record_pairs = {}
        record_pairs = []
        
        for cluster_id in np.unique(pred_clusters):
            cluster_mask = (cluster_id==pred_clusters)
            original_indices = df_numeric.index[cluster_mask].tolist()
            
            if config.verbose:
                print("original_indices",original_indices)
                print(f"KMeans Suggest add records {original_indices} to cluster {cluster_id}")
            
            for i in range(len(original_indices)):
                for j in range(i+1, len(original_indices)):
                    print(f"Record {original_indices[i]} vs {original_indices[j]} cossim = {all_records_cossim_matrix_df.loc[original_indices[i],original_indices[j]]}")
                    
            cluster_records[cluster_id] = original_indices

            #pairs for each cluster
            pairs = []
            for i in range(len(original_indices)):
                for j in range(i+1,len(original_indices)):
                    pairs.append((original_indices[i],original_indices[j]))
            cluster_record_pairs[cluster_id] = pairs
            record_pairs.extend(pairs)

        #elbow
        if target_num_k>1:
            kmeans.fit(num_features_array)
            self.wcss_dict[k] = kmeans.inertia_
            print("\nElbow within-cluster sum of squares (wcss)",self.wcss_dict)
        else:
            print("Skip Knee",target_num_k)

        #silhouette
        self.silhouette_analysis(num_features_array, pred_clusters, k)

        # Return both cluster assignments and similarity matrix
        return df_numeric, pred_clusters, cluster_records, cluster_record_pairs, record_pairs, similarity_matrix

    def silhouette_analysis(self, num_features_array, pred_clusters, n_clusters):

        if n_clusters > 1 and n_clusters < len(pred_clusters): #k must be greater than 1 and less than len data for siloutte
            
            score = silhouette_score(num_features_array, pred_clusters)
            
            if config.verbose:
                print("This K",n_clusters,"has silhouette score",score, "vs silhouette best_score",self.best_sil_score)
            
            if score > self.best_sil_score:
                self.best_sil_score = score
                self.best_sil_k = n_clusters        
                if config.verbose:
                    print("silhouette score",score,"silhouette best_k",self.best_sil_k)
        else:
            print("[WARNING] K<1 or k=len, No silhouette_score to calculate.")

        if config.verbose:
            print("silhouette best_k",self.best_sil_k)