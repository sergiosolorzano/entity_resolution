import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer, RobustScaler, QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer
from jellyfish import soundex, metaphone

import config

class Rule_Methods_Library:
    
    def _phonetic(self, feature_series: pd.Series, method_params:dict, feature_name:str) -> pd.Series:
        
        def encode(x):
            x = str(x).strip()
            key_list = []
            if "soundex" in method_params["algorithms"]:
                key_list.append(soundex(x))
            if "metaphone" in method_params["algorithms"]:
                key_list.append(metaphone(x))
            
            return "_".join(key_list)
        
        return feature_series.apply(encode)
    
    def _phonetic_combination(self, feature_series: pd.Series, method_params:dict, feature_name:str) -> pd.Series:
        
        def encode_combo(x):
            key_list = []
            if "first_char" in method_params["algorithms"]:
                key_list.append(str(x).lower()[0:1])
            
            if "first_two_char" in method_params["algorithms"]:
                key_list.append(str(x).lower()[0:2])

            if "first_three_char" in method_params["algorithms"]:
                key_list.append(str(x).lower()[0:3])

            if "last_three_char" in method_params["algorithms"]:
                key_list.append(str(x).lower()[-3:])

            if "consonants" in method_params["algorithms"]:
                key_list.append(''.join([c for c in str(x).lower() if c.isalpha() and c not in 'aeiou']))
            
            return key_list
        
        return feature_series.apply(encode_combo)
    
    def _two_of_three_date(self, feature_series:pd.Series, method_params:dict, feature_name:str) -> pd.Series:

        feature_series = pd.to_datetime(feature_series, format="%d/%m/%Y", errors="coerce")
        
        def combine(x):
            try:
                return [
                    f"my_{x.month:02d}_{x.year:02d}_dy_{x.day:02d}_{x.year:02d}",
                    f"my_{x.month:02d}_{x.year:02d}_dm_{x.day:02d}_{x.month:02d}",
                    f"my_{x.day:02d}_{x.year:02d}_dm_{x.day:02d}_{x.month:02d}"
                ]
            except (AttributeError, TypeError, ValueError) as e:
                print(f"[ERROR]: Can't process Blcking Method _one_of_three_date: {e}")
                return ["invalid_date"]
            
        result = feature_series.apply(combine)
        
        #print("_two_of_three_date key list",result)

        return result
                
    def _adaptive_binning(self, feature_series:pd.Series, method_params:dict, feature_name:str) -> pd.Series:

        global_transformers = self._load_global_transformers(feature_name)
        global_robust_scaler = global_transformers["robust_scaler"]
        global_kb_discretizer = global_transformers["kb_discretizer"]

        robust_scaled_values = global_robust_scaler.transform(feature_series.values.reshape(-1,1))
        #ravel to flatted it 1D compatible with series
        bins = global_kb_discretizer.transform(robust_scaled_values).ravel()

        if config.verbose:
            print(f"\n{feature_name} - Bin edges (after applying global transformers):")
            print("\tEdges:", global_kb_discretizer.bin_edges_[0])
            print(f"\tThere are {len(global_kb_discretizer.bin_edges_[0])} edges thus {len(global_kb_discretizer.bin_edges_[0])-1} bins")
            print("\tGlobal counts per bin (on full data):", np.bincount(global_kb_discretizer.transform(robust_scaled_values).astype(int).ravel()))

        bins_str = pd.Series(bins, index=feature_series.index).astype(str)
        
        #print("_adaptive_quantile_binning key list", bins_str)
        
        return bins_str

    def _sliding_window(self, feature_series:pd.Series, method_params:dict, feature_name:str) -> pd.Series:
        dt = pd.to_datetime(feature_series)
        days = (dt - pd.Timestamp('1970-01-01')).dt.days
        return (days // method_params["algorithms"]).astype(str)

    def _load_global_transformers(self, feature):

        transformers = {}

        def load_kb_discretizer():
            kb_discretizer_fn = f"{config.global_transformers_dir}/{feature}{config.global_kb_discretizer_fn}"
            return joblib.load(kb_discretizer_fn)

        def load_robust_scaler():
            scaler_fn = f"{config.global_transformers_dir}/{feature}{config.global_robust_scaler_fn}"
            return joblib.load(scaler_fn)
        
        try:
            disc = load_kb_discretizer()
            transformers["kb_discretizer"] = disc
        except:
            print("[ERROR]: Failed to Load Global Discretizer")

        try:
            scaler = load_robust_scaler()
            transformers["robust_scaler"] = scaler
        except:
                print("[ERROR]: Failed to Load Global Discretizer")

        return transformers

    

        



