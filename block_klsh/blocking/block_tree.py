import json, joblib
import copy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer, RobustScaler, QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer

from blocking.rule_methods import Rule_Methods_Library
import config

class BlockTree:
    class Block():
        def __init__(self):
            self.hierarchy_level = 0
            self.feature = None
            self.parent_key = None
            self.block_key = None
            self.rule = None
            self.indices = []
            self.block_leaf = None

    def __init__(self):
        self.rule_stage_info_dict = {}
        self.initial_block = config.block_tree_initial_block
        self.rule_method_instance = Rule_Methods_Library()
        
        #tree creation
        self.tree_blocks = {}
        self.tree_block_count = 0
        self.tree_hierarchy_count = 0

        #determine pair weights graph
        self.prepruned_graph = {}
        #provenance calculation
        self.prepruned_graph_provenance = {}
        self.pruned_graph = {}
        self.pruned_graph_provenance = {}
        
    def _print_all(self):
        print("== Printing All Generated Blocks: ==")
        for stage, block_dict in self.tree_blocks.items():
            for counter, (block_key, block_instance) in enumerate(block_dict.items(), start=1):
                print("Block Key",block_key)
                print("\tBlock Hierarchy Level:",block_instance.hierarchy_level)
                print("\tBlock Feature:",block_instance.feature)
                print("\tBlock Parent Key:",block_instance.parent_key)
                print("\tBlock Key:",block_instance.block_key)
                print("\tLeaf Key:",block_instance.block_leaf)
                print("\tBlock Rule:",block_instance.rule)
                print("\tBlock Record Indices:",block_instance.indices)
        print(f"== End of Tree Blocks, count: {counter} ==")

    def _print_instance(self, block_instance_key):
        for stage, block_dict in self.tree_blocks.items():
            for block_key, block_instance in block_dict.items():
                if(block_key == block_instance_key):
                    print("Block Key",block_key)
                    print("\tBlock Hierarchy Level:",block_instance.hierarchy_level)
                    print("\tBlock Feature:",block_instance.feature)
                    print("\tBlock Parent Key:",block_instance.parent_key)
                    print("\tBlock Key:",block_instance.block_key)
                    print("\tBlock Rule:",block_instance.rule)
                    print("\tBlock Record Indices:",block_instance.indices)


    def _load_blocking_rules(self, file_path=config.blocking_rules_fname):
        try:
            with open(file_path, 'r') as f:
                rules = json.load(f)['rules']
            return rules
        except FileNotFoundError:
            print(f"[ERROR]: Could not find {file_path}.")
            return {}
        
    def _register_scenario_blocking_rules(self, scenario_blocking_rules, blocking_rules_lib):

        for stage_num, (feature, rule) in enumerate(scenario_blocking_rules, 1):
            
            if isinstance(rule, dict):
            
                self.rule_stage_info_dict[stage_num] = {
                    'stage_num': stage_num,
                    'feature': feature,
                    'rule_name': rule.get('rule'),
                    'rule_specs': blocking_rules_lib[rule.get('rule')],
                }

        return self.rule_stage_info_dict
            
    def _fit_global_transformers(self, data_df):

        for stage, rule in self.rule_stage_info_dict.items():
            
            if rule["rule_name"] == "adaptive_quantile_binning" or rule["rule_name"] == "adaptive_uniform_binning":
                
                try:
                    n_bins = rule["rule_specs"]["params"]["n_bins"]
                except:
                    n_bins = config.default_bins
                    print(f"Global KBDiscretizer: {rule["rule_name"]} missing n_bin, setting default {config.default_bins}")

                values = data_df[rule["feature"]].values.reshape(-1,1)
                
                robust_scaler = RobustScaler()
                robust_scaled_values = robust_scaler.fit_transform(values)
                joblib.dump(robust_scaler, f"{config.global_transformers_dir}/{rule["feature"]}{config.global_robust_scaler_fn}")

                kb_discretizer = KBinsDiscretizer(n_bins=n_bins, encode=rule["rule_specs"]["params"]["KBinsDiscretizer_encode"], strategy=rule["rule_specs"]["params"]["KBinsDiscretizer_binning_method"])
                kb_discretizer.fit(robust_scaled_values)
                joblib.dump(kb_discretizer, f"{config.global_transformers_dir}/{rule["feature"]}{config.global_kb_discretizer_fn}")
                
                if config.verbose:
                    print(f"Done fitting and saving global KBdiscretizer/scaler for {rule["rule_name"]} feature {rule["feature"]}.")

            else:
                print(f"=== WARNING: No rule requested for Global Transformer to Fit for {rule["rule_name"]} ===")

    def _create_block_key(self, data_df):

        current_block = {}
        
        #root node
        initial_block = {config.block_tree_initial_block: frozenset(data_df.index)}

        block_instance = self.Block()
        block_instance.hierarchy_level = 0
        block_instance.rule = None
        block_instance.feature = None
        block_instance.block_key = config.block_tree_initial_block
        block_instance.block_leaf = config.block_tree_initial_block
        self.tree_block_count += 1
        block_instance.indices = sorted(frozenset(data_df.index))
        block_instance.parent_key = None
        self.tree_blocks[0] = {}
        self.tree_blocks[0][config.block_tree_initial_block] = block_instance
        current_block[0] = initial_block
        
        for stage, rule in self.rule_stage_info_dict.items():

            current_block[stage] = {}
            self.tree_blocks[stage] = {}
            
            for k,v in current_block[stage-1].items():
                parent_key = k
                subset_idx_list = list(v)
                current_block = self._add_block(parent_key, subset_idx_list, stage, rule, data_df, current_block)

        print("\n"); self._print_all()

    def _add_block(self, parent_key, subset_idx_list, stage, rule, data_df, current_block):
        
        current_block = copy.deepcopy(current_block)

        rule_method = self._rule_method_allocator(rule['rule_name'])
        if rule_method is not None:
            blocks = rule_method(data_df[rule['feature']].loc[subset_idx_list], rule['rule_specs']['params'], rule['feature'])
            # print("Blocks received\n",blocks)

            keys_df = pd.DataFrame({"index":blocks.index,"key":blocks})        
            #reassign keys_df so each record idx has its own key
            keys_df = keys_df.explode("key")

            #create sub-blocks
            for new_key, subset_df in keys_df.groupby("key"):
                block_instance = self.Block()
                keys_idxs = frozenset(subset_df.index)
                new_block_key = parent_key + f"-{rule['feature']}_{rule['rule_name']}:{new_key}"
                current_block[stage][new_block_key] = sorted(keys_idxs)

                block_instance.hierarchy_level = stage
                block_instance.rule = rule['rule_name']
                block_instance.feature = rule['feature']
                block_instance.block_key = new_block_key
                block_instance.block_leaf = f"{rule['rule_name']}:{new_key}"
                block_instance.indices = sorted(keys_idxs)
                block_instance.parent_key = parent_key
                self.tree_blocks[stage][new_block_key] = block_instance
                self.tree_hierarchy_count = stage

        return current_block

    def _rule_method_allocator(self, block_rule_name):
        # print("Looking for ",block_rule_name)
        if block_rule_name=="phonetic":
            return self.rule_method_instance._phonetic
        if block_rule_name=="phonetic_combination":
            return self.rule_method_instance._phonetic_combination
        if block_rule_name=="one_of_three_date":
            return self.rule_method_instance._one_of_three_date
        if block_rule_name=="two_of_three_date":
            return self.rule_method_instance._two_of_three_date
        if block_rule_name=="adaptive_quantile_binning":
            return self.rule_method_instance._adaptive_binning
        if block_rule_name=="adaptive_uniform_binning":
            return self.rule_method_instance._adaptive_binning
        if block_rule_name=="residual_merge":
            return self.rule_method_instance._residual_merge

    def _track_pair_provenance_and_weights(self):
        
        records =  set()

        #create graph: create set with all record indices
        for stage, block_dict in self.tree_blocks.items():
            for block_key, block_instance in block_dict.items():
                if block_instance.block_key == config.block_tree_initial_block:
                    continue
                for record_idx in block_instance.indices:
                    records.update([record_idx])
        #now create graph for each record idx
        for record_idx in records:
            self.prepruned_graph[record_idx] = {}

        for stage, block_dict in self.tree_blocks.items():
            for key, block_instance in block_dict.items():

                #iterate over the block instances without repeat
                for i in range(0, len(block_instance.indices)):#start at 1 because 0 is root
                    for j in range(i+1, len(block_instance.indices)):

                        if block_instance.hierarchy_level == 0:
                            continue

                        #add pair key provenance
                        edge_pair = (min(block_instance.indices[i], block_instance.indices[j]), 
                                    max(block_instance.indices[i], block_instance.indices[j]))
                        
                        if edge_pair not in self.prepruned_graph_provenance:
                            self.prepruned_graph_provenance[edge_pair] = set()
                        
                        self.prepruned_graph_provenance[edge_pair].update([block_instance.block_key])
                        
                        #add +1 to weight count of pair co-ocurrences=> e.g. graph_weights dict (record0, {record1:co-ocurrences})
                        self.prepruned_graph[block_instance.indices[i]][block_instance.indices[j]] = self.prepruned_graph[block_instance.indices[i]].get(block_instance.indices[j], 0) + 1
                        self.prepruned_graph[block_instance.indices[j]][block_instance.indices[i]] = self.prepruned_graph[block_instance.indices[j]].get(block_instance.indices[i], 0) + 1

        print("\n");print("self.prepruned_graph",self.prepruned_graph)

        return self.prepruned_graph
    
    def _prune_graph(self):

        pruned_pairs = []

        #retain singletons in pre-pruned graph
        self.pruned_graph = {record:{} for record in self.prepruned_graph}

        #iterate over connections of prepruned G nodes and assigns edge weight if above required weight
        for record1, connection in self.prepruned_graph.items():
            for record2, pair_weight in connection.items():
                if pair_weight > config.static_threshold_weight:
                    self.pruned_graph[record1][record2] = pair_weight
                    self.pruned_graph[record2][record1] = pair_weight

        for record1, connection in self.pruned_graph.items():
            for record2, weight in connection.items():
                if (record1,record2) in self.prepruned_graph_provenance:
                    pruned_pairs.append((record1,record2))
                    self.pruned_graph_provenance[(record1,record2)] = self.prepruned_graph_provenance[(record1,record2)]

        print("\n");print("self.pruned_graph", self.pruned_graph)

        return pruned_pairs