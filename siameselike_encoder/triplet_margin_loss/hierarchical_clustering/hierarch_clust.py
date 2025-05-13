import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import torch

torch.manual_seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.logger import Tne_Logger

import config

class Hierarchical_Clustering:
    def __init__(self, logger, device):
        self.logger = logger
        self.device=device

    def hierarchical_cluster_embeddings(self, embeddings, n_clusters=None, distance_threshold=None, linkage='ward'):
        
        if n_clusters is None and distance_threshold is None:
            distance_threshold = None
            n_clusters = 3
        
        #init and fit clustering model
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            linkage=linkage
        )
        
        model = model.fit(embeddings)

        self.logger.info(f"Hierarchical Cluster assignments: {model.labels_}")
        
        return model    