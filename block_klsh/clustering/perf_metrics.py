from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

import config

class Performance_Metrics:

    def calculate_metrics(self, candidate_pairs, true_pairs, k, metrics_pairs_result: list):
        
        tp = len(set(candidate_pairs) & set(true_pairs))
        fp = len(set(candidate_pairs) - set(true_pairs))
        fn = len(set(true_pairs) - set(candidate_pairs))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision*recall)/(precision+recall) if (precision*recall)>0 else 0

        metrics_pairs_result.append({
                    "K":k, 
                    "F1": f1, 
                    "Precision": precision, 
                    "Recall": recall
                    })

        return metrics_pairs_result
    
    def ssilhouette(self, num_features_array, labels, k_top):
        
        X = num_features_array  # X is your 2D array of features

        best_k = None
        best_score = -1
        for k in range(2, k_top):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            
            if config.verbose:
                print(f"k={k}, silhouette score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_k = k

        if config.verbose:
            print("Best k chosen based on silhouette score:", best_k)
    
    def find_best_elbok_k(self, wcss_dict: dict):
        
        sorted_k = sorted(wcss_dict.keys())
        sorted_wcss = [wcss_dict[k] for k in sorted_k]
        
        if config.verbose:
            print("Sorted WCSS:", sorted_wcss)
        return KneeLocator(
            sorted_k,
            sorted_wcss,
            curve='convex', 
            direction='decreasing',
            S=0.3
        ).knee
