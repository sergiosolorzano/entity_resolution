import matplotlib.pyplot as plt

import config

def plot_elbow(wcss_dict: dict, component_idx):
    
    plt.figure(figsize=(8,5))
    k_range = [k for k in wcss_dict.keys()]
    wcss_range = [v for v in wcss_dict.values()]
    
    plt.plot(k_range, wcss_range, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('within-cluster sum of squares - WCSS (Inertia)')
    plt.title(f"Elbow Method KLSH Component {component_idx} for Optimal K")
    plt.savefig(f"{config.graph_dir}/klsh_elbow_component{component_idx}", dpi=300, bbox_inches='tight', pad_inches=0.5) 
    plt.close()
