import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer, RobustScaler, QuantileTransformer
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
                
    def _sliding_window(self, feature_series:pd.Series, method_params:dict, feature_name:str) -> pd.Series:
        dt = pd.to_datetime(feature_series)
        days = (dt - pd.Timestamp('1970-01-01')).dt.days
        return (days // method_params["algorithms"]).astype(str)