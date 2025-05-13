import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import config

class Data_Path_Provider:
    def __init__(self, logger):
        self.logger = logger
        self.config = config.inputs_config

    def get_paths(self, mode, is_training):
        
        if mode not in ['tne_mode']:
            raise ValueError("Invalid mode. Expected 'tne_mode'.")

        if mode == 'tne_mode':
            if config.run_inference_hierarchical_clustering:
                data_key = 'tne_infer_data_csv_fpath'
                labels_key = "tne_infer_labels_csv_fpath"
            else:
                data_key = 'tne_train_data_csv_fpath' if is_training else 'tne_eval_data_csv_fpath'
                labels_key = 'tne_train_labels_csv_fpath' if is_training else 'tne_eval_labels_csv_fpath'

        data_path = self.config.get('paths', {}).get(data_key, None)
        labels_path = self.config.get('paths', {}).get(labels_key, None)

        if not data_path or not labels_path:
            raise ValueError(f"Paths for {mode} ({'training' if is_training else 'evaluation'}) are missing in the configuration.")

        return {
            'data_path': data_path,
            'labels_path': labels_path
        }