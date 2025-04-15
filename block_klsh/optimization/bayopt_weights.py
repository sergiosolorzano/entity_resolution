from skopt import gp_minimize
import numpy as np
from skopt.callbacks import EarlyStopper

from context import ER_Context
import config

#custom skopt value stopper
class CustomEarlyStopper(EarlyStopper):
    def __init__(self, early_stop_trigger):
        self.stop_iter_target_value = early_stop_trigger

    def _criterion(self, bayes_result):
        if bayes_result.fun == self.stop_iter_target_value:
            return True

class BayOpt:
    def __init__(self, component_pairs_and_singletonlist, data_df, k_bottom, k_top):
        self.component_pairs_and_singletonlist = component_pairs_and_singletonlist
        self.data_df = data_df
        self.k_bottom = k_bottom
        self.k_top = k_top
        self.iteration = 0

    def run_klsh_bayopt(self, ctx: ER_Context):
        
        search_space = [(0.0, 1.0) for i in range(len(config.feature_weights))]
        
        early_stopper = CustomEarlyStopper(config.bayes_early_stopper_value)
        result = gp_minimize(
        func =lambda weights: self.objective_bayopt(weights, ctx),
        dimensions=search_space, n_calls=config.bayes_n_calls, random_state=42, callback=[early_stopper]
        )
        
        #average those with some top result
        top_weights = []
        top_score = []

        for i, weights in enumerate(result.x_iters):
            score = -result.func_vals[i]
            if score >= 0.99 * -(result.fun):
                top_weights.append(weights)
                top_score.append(score)

        if top_weights and len(top_weights)>1:
            avg_weights = np.mean(top_weights, axis=0)
            print("The Bayesian weights are an average of ",len(top_weights),"results.")
        else:
            avg_weights = result.x
      
        return avg_weights

    def objective_bayopt(self, feature_weights, ctx: ER_Context):

        self.iteration += 1

        config.feature_weights = {
            'tension_adj_cos': feature_weights[0],
            'tension_adj_sin': feature_weights[1],
            'tension': feature_weights[2],
            'resonance': feature_weights[3],
            'longevity_cos': config.best_weights[4],
            'longevity_sin': config.best_weights[5],
            'quality_cos': feature_weights[6],
            'quality_sin': feature_weights[7],
            'amt_sold': feature_weights[8]
        }
        
        all_components_f1 = []
        all_components_precision = []
        all_components_recall = []

        for component_idx, component_pairs in enumerate(self.component_pairs_and_singletonlist):
            
            if len(component_pairs) > 1:
                df_numeric, pred_clusters, metrics_pairs_result_df, lowest_k_highest_F1_metrics = ctx.klsh_engine_instance.predict_klsh_clusters_and_metrics(self.component_pairs_and_singletonlist, self.data_df, component_idx, self.k_bottom, self.k_top)
                
                if config.verbose:
                    print("Returning lowest_k_highest_F1_metrics",lowest_k_highest_F1_metrics['F1'], "type",type(lowest_k_highest_F1_metrics))
                
                all_components_f1.append(float(lowest_k_highest_F1_metrics['F1']))
                all_components_precision.append(float(lowest_k_highest_F1_metrics['Precision']))
                all_components_recall.append(float(lowest_k_highest_F1_metrics['Recall']))
                print(f"Finished {component_idx} Component Pairs {component_pairs}: -lowest_k_highest_F1", -lowest_k_highest_F1_metrics['F1'])
            else:
                print(f"[WARNING] Bayes: Skipped component {component_idx} as there are no pairs: len {len(component_pairs)}")
        
        components_avg_f1 = np.mean(all_components_f1)
        components_avg_precision = np.mean(all_components_precision)
        components_avg_recall = np.mean(all_components_recall)
        
        if config.verbose:
            print(f"End Bayes Iter {self.iteration}:")
            print("\tIter Current Weights",feature_weights)
            print("\tIter Average F1", components_avg_f1)
            print("\tIter Average Precision", components_avg_precision)
            print("\tIter Average Recall", components_avg_recall)

        return -components_avg_f1
        
        