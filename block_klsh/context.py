from clustering.klsh_engine import KLSH_Engine
from features.features_engineering import Feature_Engineering
from clustering.perf_metrics import Performance_Metrics


class ER_Context:
    def __init__(self):
        self.feature_engineering_instance = Feature_Engineering()
        self.perf_metrics_instance = Performance_Metrics()
        self.klsh_engine_instance = KLSH_Engine(self.feature_engineering_instance, self.perf_metrics_instance)
