# About
This directory holds the model and scripts for model training with triplet margin loss criterion and hard-mining sampling.

# Usage
Model Configuration:
- Training (Evaluation occurs at epoch intervals) toggled at config.run_inference_hierarchical_clustering = False
- Set number of training epochs at config.tne_train_epoch = #
- Set loss function at config.loss_function = choose contrastive" "tripletmargin" or "tripletmargin_mining"
- To run Inference on a test dataset set config.run_inference_hierarchical_clustering = True

Execution:

1. Execute the program
	$ python m_manager.py
2. Results:
- logs shown on terminal
- plots directory/
	- distance_distrib_Train_e_X.png shows training embeddings anchor-positive and anchor-negative distance distributions
	- distance_distrib_Eval_e_X.png shows eval embeddings anchor-positive and anchor-negative distance distributions
	- training loss evolution plot
	- auroc and confusion matrix plots
	- cvs with eval and training metrics