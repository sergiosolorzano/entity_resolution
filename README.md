This repo includes a set of experiments with different architectures to explore their capability in [entity resolution](https://en.wikipedia.org/wiki/Record_linkage) problems to solve deduplication and map diverse records to their corresponding entities.

The datasets for these experiments are models of piano brands.

The project is broken down by directory, with each directory representing a different approach:

1. [block_klsh](https://github.com/sergiosolorzano/entity_resolution/tree/main/block_klsh): An entity resolution experiment based on Meta-Blocking and KLSH: The proposed approach has three sequential stages. 
    - Firstly the generation of a hierarchical graph for records in blocks and according to blocking rules. 
    - The resulting record pair relationships enable the construction of a record relationship grapth with components. 
    - Finally we cluster the records for each component using a K-Means algorithm, an approach we refer herein and based on [existing literature](https://arxiv.org/pdf/1810.05497), as KLSH.
    - The methodology and its implementation is described in more detail in this publication ["Entity Resolution: Meta-Blocking and KLSH"](https://app.readytensor.ai/publications/entity-resolution-meta-blocking-and-klsh-3hz55CPSvHPs).
    - the code for the project is [here](https://github.com/sergiosolorzano/entity_resolution/tree/main/block_klsh)

2. [siameselike_encoder](https://github.com/sergiosolorzano/entity_resolution/tree/main/siameselike_encoder): An entity resolution experiment based on an Encoder and Clustering algorithm.
    - We create and train an encoder/network with transformed entity tabular feature data.
    - A feed forward network trained on tabular data learns representations from record features that serve as inputs for a clustering algorithm to separate entities in the representation space. 
    - The methodology and its implementation is described in more detail in this publication ["Entity Resolution: Learned Representations of Tabular Data with Classic Neural Networks"](https://app.readytensor.ai/publications/entity-resolution-learned-representations-of-tabular-data-with-classic-neural-networks-MtUrsAPP6Mdt).
    - the code for the project is [here](https://github.com/sergiosolorzano/entity_resolution/tree/main/siameselike_encoder)

<p>&nbsp;</p>

## License
This project is licensed under the MIT License. See LICENSE.txt for more information.

<p>&nbsp;</p>

## Contact
For questions or collaborations please reach out to sergiosolorzano@gmail.com

<p>&nbsp;</p>

If you find this helpful you can buy me a coffee :)

<a href="https://www.buymeacoffee.com/sergiosolorzano" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>      
