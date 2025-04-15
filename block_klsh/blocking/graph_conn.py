import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

import config

class DeduP_Graph():
    def __init__(self):
        self.graph = None

    def _create_graph(self, block_tree: dict):
        G = nx.DiGraph()

        for hierarchy_level, tree_block_dict in block_tree.items():
            for block_key, block_instance in tree_block_dict.items():
                G.add_node(
                    block_key,
                    hierarchy_level = block_instance.hierarchy_level,
                    feature = block_instance.feature,
                    parent_key = block_instance.parent_key,
                    block_key = block_instance.block_key,
                    block_leaf = block_instance.block_leaf,
                    rule = block_instance.rule,
                    indices = block_instance.indices,
                    node_labels = f"{block_instance.block_leaf}\n{len(block_instance.indices)} records",
                    block_instance = block_instance
                )
        
        self.graph = G

    def _create_edges(self, block_tree: dict):
        edges_from_to_key = []
        for hierarchy_level, tree_block_dict in block_tree.items():
            for block_key, block_instance in tree_block_dict.items():
                if block_instance.parent_key == None:
                    continue
                edges_from_to_key.append((block_instance.parent_key, block_key))

        self.graph.add_edges_from(edges_from_to_key)

    def _visualize_graph_tree(self):
        #create a mapping of original node names to integers
        mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        G_int = nx.relabel_nodes(self.graph, mapping)
    
        #graphviz layout 'dot' style to arrange nodes left-to-right
        pos = graphviz_layout(G_int, prog='dot', args='-Grankdir=LR')

        #map positions back to original nodes
        pos_original = {original: pos[mapped] for original, mapped in mapping.items()}

        #get node labels that show block_leaf value and record count
        leave_names = nx.get_node_attributes(self.graph, "indices")

        plt.figure(figsize=(12, 6))
        nx.draw(self.graph, 
                pos_original,
                with_labels=True, 
                labels=leave_names, 
                font_size=5,
                arrows=True, 
                node_size=500,
                node_color="lightblue")
        
        #draw labels separately with custom color
        nx.draw_networkx_labels(self.graph, 
                            pos_original, 
                            labels=leave_names, 
                            font_size=5,
                            font_color='red')  # Set the color to red
        
        plt.title("Hierarchical Directed Graph")
        plt.savefig(f"{config.graph_dir}/graph_tree", dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close()