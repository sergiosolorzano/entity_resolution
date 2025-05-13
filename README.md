This repo includes a set of experiments with different architectures to explore their capability in solving deduplication and mapping diverse records to their correct entities.

The datasets for these experiments are models of piano brands.

The project is broken down by directory, with each directory representing a different approach:

1. [block_klsh](https://github.com/sergiosolorzano/entity_resolution/tree/main/block_klsh): An entity resolution experiment based on Meta-Blocking and KLSH:
- It builds a hierarchical directed graph where each level corresponds to nodes generated from blocking rules. This reduces the number of comparison to O(n).
- Records in nodes can be grouped together into components subject to a minimum number of co-ocurrences within the graph.
- The resulting records in a component are broken down into clusters applying KMeans, referred as KLSH. 
- The methodology and its implementation is described in more detail in this blog ["Entity Resolution: Meta-Blocking and KLSH"](https://app.readytensor.ai/publications/entity-resolution-meta-blocking-and-klsh-3hz55CPSvHPs).

2. [siameselike_encoder](https://github.com/sergiosolorzano/entity_resolution/tree/main/siameselike_encoder): An entity resolution experiment based on an Encoder and Clustering algorithm.
- We create and train an encoder/network with entity tabular feature data.
- A feed forward network trained on tabular data learns representations from record features that serve as inputs for a clustering algorithm to separate entities in the representation space. 
- The methodology and its implementation is described in more detail in this blog ["Entity Resolution: Learned Representations of Tabular Data with Classic Neural Networks"](https://app.readytensor.ai/publications/entity-resolution-learned-representations-of-tabular-data-with-classic-neural-networks-MtUrsAPP6Mdt).
