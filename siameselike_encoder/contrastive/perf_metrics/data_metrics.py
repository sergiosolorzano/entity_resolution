import os, sys
import pandas as pd
import matplotlib as plt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from features.feature_engineering import Feature_Engineering
    from perf_metrics.plots import Plots
    from utils.logger import Tne_Logger

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CURRENT_DIR)

class Data_Metrics:
    def __init__(self, feature_engineering_instance: 'Feature_Engineering',
                 plots_instance: 'Plots',
                 logger: 'Tne_Logger'):
        self.feature_engineering_instance = feature_engineering_instance
        self.plots_instance = plots_instance
        self.logger = logger

    def compute_statistics(self, df, role):
        stats = {}
        features = ['tension', 'resonance', 'quality', 'longevity_cos', 'longevity_sin']
        for feature in features:
            if feature in df.columns:
                data = df[feature]
                stats[feature] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'iqr': data.quantile(0.75) - data.quantile(0.25)
                }

                #find outliers iqr
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                stats[feature]['outlier_percentage'] = len(outliers) / len(data) * 100

        self.logger.info(f"\nStatistics for {role}:")
        self.logger.info(pd.DataFrame(stats).transpose())

        return stats

    def compare_data_original(self, csv_train_data_file_path, csv_train_labels_file_path,
                            csv_eval_data_file_path, csv_eval_labels_file_path, title):
        
        df_train_data, train_labels_flattened,train_labels_df  = self.feature_engineering_instance.get_raw_data(csv_train_data_file_path, csv_train_labels_file_path)
        df_eval_data, eval_labels_flattened, eval_labels_df = self.feature_engineering_instance.get_raw_data(csv_eval_data_file_path, csv_eval_labels_file_path)

        self.compare_data(df_train_data, train_labels_df, df_eval_data, eval_labels_df, "original")

    def compare_data(self, train_data, train_labels, eval_data, eval_labels, title):

        self.logger.info(f"train data {train_data.head()}")
        self.logger.info(f"eval data {eval_data.head()}")

        # Analyze Evaluation Data
        eval_stats = {}
        for role, indices in zip(['anchor', 'positive', 'negative'], ['anchor_piano', 'positive_piano', 'negative_piano']):
            role_indices = eval_labels[indices].values.flatten()
            role_data = eval_data.iloc[role_indices]
            eval_stats[role] = pd.DataFrame(self.compute_statistics(role_data, f"Evaluation {role}")).transpose()

        # Analyze Training Data
        train_stats = {}
        for role, indices in zip(['anchor', 'positive', 'negative'], ['anchor_piano', 'positive_piano', 'negative_piano']):
            role_indices = train_labels[indices].values.flatten()
            role_data = train_data.iloc[role_indices]
            train_stats[role] = pd.DataFrame(self.compute_statistics(role_data, f"Training {role}")).transpose()

        for role in ['anchor', 'positive', 'negative']:
            self.plots_instance.plot_combined_statistics(train_stats[role], eval_stats[role], role, title)