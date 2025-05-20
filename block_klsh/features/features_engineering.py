import numpy as np
import pandas as pd
import hashlib

import config

class Feature_Engineering:
   
    def embed_bool_category(self, bool_col: pd.Series):

        def quarter_circle_encoding(value):
            if value==0:
                return 1.0, 0.0
            else:
                return 0.0, 1.00
            
        projection = bool_col.apply(quarter_circle_encoding)

        return projection.map(lambda x: x[0]), projection.map(lambda x: x[1])

    def embed_ordinal_category(self, quality_col: pd.Series):

        full_quality_range_list = list(range(config.quality_ordinal_range[1]+1))
        quality_mappings = {value: value for value in full_quality_range_list}

        df = pd.DataFrame()
        df['quality_ordinal'] = quality_col.map(quality_mappings)

        def quarter_circle_encoding(ordinal_value, max=config.quality_ordinal_range[1]):

            angle = ordinal_value / max * (np.pi/2)
            return np.cos(angle), np.sin(angle)
        
        projection = df['quality_ordinal'].apply(quarter_circle_encoding)

        return projection.map(lambda x: x[0]), projection.map(lambda x: x[1])
    
    def quarter_circle_dt_projection_series(self, date_series):
    
        max_seconds = config.time_max_time_frame.total_seconds()

        date_series = pd.to_datetime(date_series, errors="coerce")

        def project_date(date):
            if pd.isna(date):
                print("ERROR DATE normalization",date)
                exit()
            delta = (date - config.time_reference_date).total_seconds()
            fraction = max(0, min(1, delta / max_seconds))
            theta = fraction * (np.pi / 2)
            return np.cos(theta), np.sin(theta)

        projection = date_series.apply(project_date)

        cos_series = projection.map(lambda x: x[0])
        sin_series = projection.map(lambda x: x[1])
        
        return cos_series, sin_series
    
    #two_of_three_date binning
    def get_date_features(self, date_series):
        date_series = pd.to_datetime(date_series, errors="coerce")
        
        #create binary features for each date component pair
        my_features = np.zeros(len(date_series))  # month-year
        dy_features = np.zeros(len(date_series))  # day-year
        dm_features = np.zeros(len(date_series))  # day-month
        
        for i, date in enumerate(date_series):
            if pd.notna(date):
                def deterministic_hash(s):
                    return int(hashlib.md5(f"seed42_{s}".encode()).hexdigest(), 16)
                #create hash values for each component pair
                my_hash = deterministic_hash(f"{date.month:02d}_{date.year:04d}") % 1000
                dy_hash = deterministic_hash(f"{date.day:02d}_{date.year:04d}") % 1000
                dm_hash = deterministic_hash(f"{date.day:02d}_{date.month:02d}") % 1000
                
                # Normalize to 0-1 range
                my_features[i] = my_hash / 1000
                dy_features[i] = dy_hash / 1000
                dm_features[i] = dm_hash / 1000
        
        return np.column_stack([my_features, dy_features, dm_features])