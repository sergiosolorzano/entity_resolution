import pandas as pd
import ast
import networkx as nx
import matplotlib.pyplot as plt

import config as config

def print_candidates_for_each_block(blocks):
    candidate_pair_indices = set()
    
    for block_key, block in blocks.items():
        #skip initial_block and blocks with fewer than 2 records
        if block_key == config.block_tree_initial_block or len(block) < 2:
            continue
            
        indices = list(block)
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                candidate_pair_indices.add((indices[i], indices[j]))
                if config.verbose:
                    print(f"Candidate pair indices in block {block_key}: {indices[i]} {indices[j]}")
    
    return candidate_pair_indices