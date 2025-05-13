import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import config

np.random.seed(42)

class Feature_Engineering:
    def __init__(self, logger):
        self.logger = logger

    def show_scaler_stored(self, scaler, feature):
        if isinstance(scaler, MinMaxScaler):
            self.logger.info(f"Scaler for {feature} (MinMaxScaler):")
            self.logger.info(f"  Min: {scaler.data_min_}")
            self.logger.info(f"  Max: {scaler.data_max_}")
        elif isinstance(scaler, StandardScaler):
            self.logger.info(f"Scaler for {feature} (StandardScaler):")
            self.logger.info(f"  Mean: {scaler.mean_}")
            self.logger.info(f"  Std (scale_): {scaler.scale_}")
        else:
            self.logger.info(f"Scaler for {feature}: Unknown scaler type ({type(scaler)}).")

    def add_noise(self, X):
        return X + np.random.normal(0, 0.05, X.shape)

    #project date to quarter circle
    def quarter_circle_projection_series(self, date_series):
        
        max_seconds = config.time_max_time_frame.total_seconds()

        date_series = pd.to_datetime(date_series, errors="coerce")

        def project_date(date):
            if pd.isna(date):
                self.logger.info(f"ERROR DATE normalization {date}")
                exit()
            delta = (date - config.time_reference_date).total_seconds()
            fraction = max(0, min(1, delta / max_seconds))
            theta = fraction * (np.pi / 2)
            return np.cos(theta), np.sin(theta)

        projection = date_series.apply(project_date)

        cos_series = projection.map(lambda x: x[0])
        sin_series = projection.map(lambda x: x[1])

        return cos_series, sin_series

    def create_pipeline(self, scaler_type="Standard"):
        
        if scaler_type == "MinMax":
            return Pipeline([
                ('minmax_constraint', MinMaxScaler(feature_range=(-1, 1))),
                ('noise_injection', FunctionTransformer(self.add_noise))
            ])
        
        if scaler_type == "Standard":
            return Pipeline([
                ('standard_scaler', StandardScaler())
            ])
        
    def apply_scaler(self, df, feature, scaler_type, train):
        
        pipeline_path = os.path.join(config.SCALER_DIR, f"{feature}_scaler.pkl")
        pipeline = self.create_pipeline(scaler_type)

        if train:
            transformed_feature = pipeline.fit_transform(df[[feature]]).flatten()
            joblib.dump(pipeline, pipeline_path)
            scaler = pipeline.named_steps.get('minmax_constraint', None)
            if scaler is None:
                scaler = pipeline.named_steps.get('standard_scaler', None)
            self.logger.info(f"Writing {feature} Scaler:")
        else:
            pipeline = joblib.load(pipeline_path)
            scaler = pipeline.named_steps.get('minmax_constraint', None)
            if scaler is None:
                scaler = pipeline.named_steps.get('standard_scaler', None)
            self.logger.info(f"Reading {feature} Scaler:")
            transformed_feature = pipeline.transform(df[[feature]]).flatten()

        #print scaler attributes
        if hasattr(scaler, 'mean_'):
            self.logger.info(f"{feature} Scaler Mean: {scaler.mean_}")
            self.logger.info(f"{feature} Scaler Std: {scaler.scale_}")
        elif hasattr(scaler, 'data_min_'):
            self.logger.info(f"{feature} Scaler Data Min: {scaler.data_min_}")
            self.logger.info(f"{feature} Scaler Data Max: {scaler.data_max_}")
        else:
            self.logger.info(f"{feature} Scaler: No attributes foud for this scaler")

        if scaler_type == "Standard":
            scaler = pipeline.named_steps["standard_scaler"]
            self.show_scaler_stored(scaler, feature)
        elif scaler_type == "MinMax":
            scaler = pipeline.named_steps["minmax_constraint"]
            self.show_scaler_stored(scaler, feature)

        return transformed_feature

    def get_raw_data(self, csv_data_file_path, csv_labels_file_path):

        labels_flattened = None
        self.logger.info(f"Reading file to normalize: {csv_data_file_path}")
        df_data = pd.read_csv(csv_data_file_path)
        self.logger.info(f"len data {len(df_data)}")
        
        labels_df = None
        if csv_labels_file_path is not None:
            labels_df = pd.read_csv(
                csv_labels_file_path
            )
            labels_flattened = labels_df.values.flatten()

        #convert longevity to datetime
        df_data["longevity"] = pd.to_datetime(df_data["longevity"], format="%d/%m/%Y", errors="coerce")

        return df_data, labels_flattened, labels_df

    def normalize_numeric_data(self, csv_data_file_path, feature_types, csv_labels_file_path, isTraining):

        labels_filtered = None
        train = isTraining
        
        df_data, labels_flattened, labels_df = self.get_raw_data(csv_data_file_path, csv_labels_file_path)
        
        #outlier detection
        for feature, _ in feature_types.items():
            if feature in ["resonance", "tension", "longevity"]:
                self.show_outliers(df_data, feature)
        
        labels_filtered = labels_df
        self.logger.info("== Ignored Remove Outliers ==")
            
        transformed_data = {}

        for feature, _ in feature_types.items():
            #print("Checking feature:", feature)

            #norm longevity
            if np.issubdtype(df_data[feature].dtype, np.datetime64):
                long_cos, long_sin = self.quarter_circle_projection_series(df_data[feature])
                transformed_data[f"{feature}_cos"] = long_cos.values.flatten()
                transformed_data[f"{feature}_sin"] = long_sin.values.flatten()
            
            #norm quality
            elif feature == "quality":
                transformed_data[feature] = df_data[feature]
            
            #norm resonance and tension
            elif feature in ["resonance", "tension"]:
                scaler_type = config.resonance_scaler if feature == "resonance" else config.tension_scaler
                transformed_data[feature] = self.apply_scaler(df_data, feature, scaler_type, train)
                
            else:
                self.logger.info(f"\n*** No Transformation Applied to {feature} ***\n")
                transformed_data[feature] = df_data[feature]

        return pd.DataFrame(transformed_data), labels_filtered

    def show_outliers(self, df, feature):
        
        self.logger.info("\n")
        self.logger.info(f"Show Outliers: Checking feature: {feature}")
        
        #compute irq-boundaries
        Q1 = df[feature].quantile(0.15)
        Q3 = df[feature].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        self.logger.info(f"feature {feature} lower bound {lower_bound} upper bound {upper_bound}")
        
        #identify outliers
        outliers_mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_rows = df[outliers_mask].copy()
        df = df.copy()
        df['triplet_index'] = df.index // 3  
        outlier_triplets = df.loc[outliers_mask, 'triplet_index'].unique()

        #display outlier
        self.logger.info(f"Outlier rows found ({len(outlier_rows)}):")
        self.logger.info(outlier_rows)