from datetime import datetime, timedelta

verbose = True

#scenario
uniform_weights_klsh = True
bayesian_optimization_klsh = False
test_optimized_weights_klsh = False
component_idx = 0 #does not apply to Bayes. Bayes calculates mean objective of all components
k_bottom = 1
k_top = 10

#feature engineering
quality_ordinal_range = (0,9) #assumes start idx 0
time_reference_date = datetime.strptime("1/1/2015", "%d/%m/%Y")
time_max_time_frame = timedelta(days=365* 60)

#blocking
blocking_rules_fname = "blocking/blocking_rules_lib.json"
block_tree_initial_block = "initial_block"
source_data = "data/example_apollo_0.csv"
graph_dir = "graphs"
static_threshold_weight = 1.5

#blocking rules
scenario_blocking_rules = ([
        ('name', {'rule':'phonetic_combination'}),
        #('resonance', {'rule': 'adaptive_quantile_binning', 'params': {'n_bins': 2}}),
    ])
default_bins = 2
global_kb_discretizer_fn = "_global_kb_discretizer.joblib"
global_robust_scaler_fn = "_global_robust_scaler.joblib"
global_transformers_dir = "global_transf"

#bayes weights opt
bayes_early_stopper_value = -1
bayes_n_calls = 100
best_weights = [0.32922948, 0.36130566, 0.20008195, 0.82066852, 0.44855293, 0.62657605, 0.36378109, 0.4405338, 0.2413675]

feature_weights = {
        'tension_adj_cos': 1,
        'tension_adj_sin': 1,
        'tension': 1,
        'resonance': 1,
        'longevity_cos': 1,
        'longevity_sin': 1,
        'quality_cos': 1,
        'quality_sin': 1,
        'amt_sold': 1
    }

##ground truth dataset pairs for performance metrics
true_pairs_component_0 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                        (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
true_pairs_component_1 = [(10, 11), (10, 12), (10, 13), (10, 15), (11, 12), (11, 13), (11, 15), (12, 13), (12, 15), (13, 15)]
true_pairs_component_2 = [(16, 17), (16, 18), (16, 19), (16, 20), (17, 18), (17, 19), (17, 20), (18, 19), (18, 20), (19, 20)]