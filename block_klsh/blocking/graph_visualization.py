import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_graph_components(graph_weights, save_fname, title, edge_provenance):

    plt.figure(figsize=(12,6))
    G = nx.Graph()

    #add nodes
    for node1, node2_weight in graph_weights.items():
        G.add_node(node1)

    #add edges
    nodes = []
    for node1, node2_weight in graph_weights.items():
        for node2, weight in node2_weight.items():
            G.add_edge(node1, node2, weight=weight)
            if (node1, node2) in edge_provenance:
                G.edges[node1, node2]['provenance'] = ", ".join(edge_provenance[(node1, node2)])

    #layout spacing
    pos = nx.spring_layout(G, seed=42, k=1.5)

    #draw nodes first
    nx.draw_networkx_nodes(G,pos, node_size=1000, node_color='lightblue')
    nodes = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=nodes, font_size=12, font_color='black')

    #add weighted weights
    for (node1, node2, data) in G.edges(data=True):
        width = data["weight"] * 2
        nx.draw_networkx_edges(G, pos,width=width, edgelist=[(node1,node2)], alpha=0.7, edge_color='navy')# edgelist = (node1,node2), 

    #choose to show provenance or weights
    #edge_labels = nx.get_edge_attributes(G, name="provenance")
    edge_labels = nx.get_edge_attributes(G, name="weight")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_weight='bold')

    plt.title(title, fontsize=20, fontweight=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    #find connected components
    components = list(nx.connected_components(G))
    
    #get pairs and singletons for each component
    singleton_list = []
    component_pairs_and_singletons = []
    for component in components:
        component = list(component)
        if len(component) == 1:
            print("\t=> singleton",component)
            singleton_list.append(component)
            component_pairs_and_singletons.append(component)
        else:
            pairs = [(component[i], component[j]) 
                    for i in range(len(component)) 
                    for j in range(i+1, len(component))]
            component_pairs_and_singletons.append(pairs)

    return G, component_pairs_and_singletons, singleton_list

def visualize_weights_graph(pruned_graph, save_fname, title):

    plt.figure(figsize=(12,6))
    G = nx.Graph()

    #add nodes
    for node_1_2, weight in pruned_graph.items():
        G.add_node(node_1_2[0])

    #add edges
    nodes = []
    for node_1_2, weight in pruned_graph.items():
        G.add_edge(node_1_2[0], node_1_2[1], weight=weight)

    #layout spacing
    pos = nx.spring_layout(G, seed=42, k=1.5)

    #draw nodes first
    nx.draw_networkx_nodes(G,pos, node_size=1000, node_color='lightblue')
    nodes = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=nodes, font_size=12, font_color='black')

    #add weighted weights
    for (node1, node2, data) in G.edges(data=True):
        width = data["weight"] * 2
        nx.draw_networkx_edges(G, pos,width=width, edgelist=[(node1,node2)], alpha=0.7, edge_color='navy')# edgelist = (node1,node2), 

    plt.title(title, fontsize=20, fontweight=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
def visualize_nx_graph(G, entities, fname, title, seed=42):
    
    plt.figure(figsize=(12,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)

    #build a mapping from node to component id
    component_map = {}
    for comp_id, comp in enumerate(entities):
        for node in comp:
            component_map[node] = comp_id

    #formatting - shift label
    pos_shifted = {}
    for node, (x, y) in pos.items():
        pos_shifted[node] = (x, y + 0.06) 

    #draw component labels in red
    nx.draw_networkx_labels(G, pos_shifted, labels=component_map, font_color="red")

    plt.title(title, pad=20)
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.5) 
    plt.close()