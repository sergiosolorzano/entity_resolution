# Entity Resolution - Meta-Blocking and KLSH

## Project:
This repo sub-directory is part of a set of projects that aim at solving deduplication and the identification of diverse records to its correct entity.

The project is an experimental prototype for research and does not address full error handling or production standards.

It builds a hierarchical directed graph where each level corresponds to nodes generated from a blocking rule/s. This approach reduces all dataset record comparisons which is an O(n²) to O(n) thus reducing the computational requirements for downstream processing since we would only be comparing records within components; the approach helps us present a reasonable scalable solution.

Each node in a graph level groups co-occurring records based on a rule.

![graph_tree](https://github.com/user-attachments/assets/0441a132-bbd2-4ea6-9dcb-880ef7c31b92)

Further hierarchical rules applied to these nodes can yield children nodes at deeper graph levels. 

Records in nodes can be grouped together into components subject to a minimum number of co-ocurrences within the graph.

![preprun_components_graph](https://github.com/user-attachments/assets/db4ebe32-50d0-4553-b5a1-ab76a256cb66)

We track block provenance so that each time two records co-occur in the same block, their edge weight increases by 1. Records whose edge weight is above a certain threshold form a graph component.

![Pruned_components_graph](https://github.com/user-attachments/assets/61b2c492-57ee-44dc-8f55-47acd16a7b29)

The resulting records in a component are broken down into clusters applying KMeans, referred herein and existing [literature](https://arxiv.org/pdf/1810.05497) as KLSH.

![klsh_entities_graph_k_4_comp_0](https://github.com/user-attachments/assets/d8266d8d-40a4-43aa-9baa-399f5306f957)

We run Bayesian optimization to determine the feature weights across all components and objective function targeting the average of all components' results F1=1.

The process results in entity resolution for the training dataset of F-1/Precision/Recall=0.952/1.00/0.916.

<p>&nbsp;</p>

## Project Blog Post:
Read the [ReadyTensor methodology blog]() for a description of the approach and its implementation.

<p>&nbsp;</p>

## Usage
1. Execute the program
```python
$ python manager.py
``` 
2. Enter the ground truth tuple pairs as a list in config.py, e.g:
```python
ground_truth_pairs_component_0 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
                        (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
```
3. Block and generate KLSH clusters by:
```python
   - set: config.uniform_weights_klsh = True
   - set a component index, e.g: config.component_idx = 0
```

4. Run bayesian optimization to find custom weights across all components:
```python
   - set: config.uniform_weights_klsh = False
   - set: config.bayesian_optimization_klsh = True
```

5. Test the optimized weights:
  - enter the bayesian optimization optimized weights:
```python
  config.best_weights = [0.32922948, 0.36130566, 0.20008195, 0.82066852, 0.44855293, 0.62657605, 0.36378109, 0.4405338, 0.2413675]
```
  - generate the updated clusters for a component based on the optimized weights:
```python
  - set: config.uniform_weights_klsh = False
  - set: config.bayesian_optimization_klsh = False
  - set: config.test_optimized_weights_klsh = True
  - set a component index, e.g: config.component_idx = 0
```

<p>&nbsp;</p>

## Dataset
A small manually produced dataset of 20 piano models that have 7 features with types including string, date, numeric and ordinal categories. It's is stored in /data.

<p>&nbsp;</p>

## Tools & Libraries
Graphs leverage networkx and pygraphviz libraries.
Flexibility is built to create blocking rules based on soundex and metaphone.

<p>&nbsp;</p>

## Installation
You can execute requirements.txt. Tested on Python 3.13.3.

pygraphviz can be tricky to install, I found this command useful:
```python
conda install -c conda-forge pygraphviz
```

<p>&nbsp;</p>

# Repo structure
<pre>
C:.
│   config.py          
│   context.py         
│   manager.py         # Main controller script  
│   requirements.txt   
│
├───blocking           # Blocking scripts and rules  
├───clustering         # KLSH clustering engine  
├───data               # Dataset  
├───features           # Feature engineering scripts  
├───global_transf      # Feature normalization artifacts  
├───graphs             # Graph images  
└───optimization       # Bayesian optimization script  
</pre>

<p>&nbsp;</p>

## License
This project is licensed under the MIT License. See LICENSE.txt for more information.

<p>&nbsp;</p>

## Contact
For questions or collaborations please reach out to sergiosolorzano@gmail.com

<p>&nbsp;</p>

If you find this helpful you can buy me a coffee :)

<a href="https://www.buymeacoffee.com/sergiosolorzano" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>      
