This repo includes a set of experiments with multiple architectures that research their capability to solve deduplication and the identification of diverse records to its correct entity.

The datasets for these experiments are models of piano brands.

The project is broken down by directory in the repo:
- [block_klsh](https://github.com/sergiosolorzano/entity_resolution/tree/main/block_klsh): An entity resolution experiment based on Meta-Blocking and KLSH:
It builds a hierarchical directed graph where each level corresponds to nodes generated from blocking rules. This reduces the number of comparison to O(n).
Records in nodes can be grouped together into components subject to a minimum number of co-ocurrences within the graph.
The resulting records in a component are broken down into clusters applying KMeans, referred as KLSH.

- tab_siamese_like_network:
We create and train an encoder/network with entity tabular feature data.
The feed forward network output of learned representations of record features for tabular data serve as inputs for a clustering algorithm to separate entities in the representation space.
