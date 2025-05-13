import os
from datetime import datetime, timedelta

###############################################################################
# DIRECTORIES / PATHS
###############################################################################
#region DIRECTORIES / PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Subdirectories relative to BASE_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
TNE_MODEL_DIR = os.path.join(BASE_DIR, "models","tne")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_PLOTS_DIR = os.path.join(BASE_DIR, "plots", "data")

###############################################################################
# FILE NAMES
###############################################################################
#region FILE NAMES

#train/eval Tabular_Numeric_Encoder
train_data_tabular_numeric_encoder_fname = "training_data_tne_triplet_nn_kaggle_dummy_largecut.csv"
train_labels_tabular_numeric_encoder_encoder_fname = "training_labels_tne_triplet_nn_kaggle_dummy_largecut.csv"
eval_data_tabular_numeric_encoder_encoder_fname = "eval_data_tne_triplet_nn_kaggle_dummy_largecut.csv"
eval_labels_tabular_numeric_encoder_encoder_fname = "eval_labels_tne_triplet_nn_kaggle_dummy_largecut.csv"

infer_data_tabular_numeric_encoder_fname = "test_data.csv"

#endregion

#model data configuration
inputs_config = {
        "numeric_features": {
            "quality":int,
            "resonance":float,
            "longevity":int,
            "tension":float
            },

        "paths":{
            "tne_train_data_csv_fpath": os.path.join(DATA_DIR, train_data_tabular_numeric_encoder_fname),
            "tne_train_labels_csv_fpath": os.path.join(DATA_DIR, train_labels_tabular_numeric_encoder_encoder_fname),
            "tne_eval_data_csv_fpath": os.path.join(DATA_DIR, eval_data_tabular_numeric_encoder_encoder_fname),
            "tne_eval_labels_csv_fpath": os.path.join(DATA_DIR, eval_labels_tabular_numeric_encoder_encoder_fname),
            "tne_infer_data_csv_fpath": os.path.join(DATA_DIR, infer_data_tabular_numeric_encoder_fname),
            "tne_infer_labels_csv_fpath": os.path.join(DATA_DIR, "None")
        }
    }

log_dataloader_triplet_idx = False
#endregion

###############################################################################
# Model MODE / FLAGS
###############################################################################
#region Model MODE / FLAGS

#Tabular_Numeric_Encoder
cuda_gpu = 0
run_inference_hierarchical_clustering = True
train_tabular_numeric_encoder = False if run_inference_hierarchical_clustering==True else True
print_training_weights_grads = False

high_embedding_weights = {
    'quality': 1,
    'resonance': 1,
    'tension': 1,
    'longevity': 1
}
#endregion

###############################################################################
# DATA LOADER / FEATURE ENGINEERING
###############################################################################
#region DATA LOADER / TRANSFORMS
tne_batch_size = 128
tne_train_eval_dataloader_shuffle = False
resonance_scaler = "MinMax"  # Options: "MinMax", "Standard"
tension_scaler = "MinMax"  # Options: "MinMax", "Standard"

time_reference_date = datetime.strptime("1/1/2010", "%d/%m/%Y")
time_max_time_frame = timedelta(days=365* 40)
#endregion

###############################################################################
# ENCODER HYPERPARAMETERS
###############################################################################
#region ENCODER HYPERPARAMETERS
# Loss Criterion
loss_function = "contrastive" #"contrastive" "tripletmargin" or "tripletmargin_mining"
contrastive_margin = 2 #embeddings are l2 norm -> cap margin to 2
contrastive_threshold = 0.7

# Scheduler
use_scheduler = True
scheduler_type = "CyclicLRWithRestarts" #if not cyclic default used=optim OneCycleLR
cyclicLRWithRestarts_restart_period = 5
cyclicLRWithRestarts_t_mult  = 1
cyclicLRWithRestarts_min_lr = 0.00001
cyclicLRWithRestarts_cyclic_policy = "cosine" #["cosine", "arccosine", "triangular", "triangular2", "exp_range"]

tne_train_epoch = 400
min_loss_break_training = 0.000001

tne_encoder_quality_num_categories = 10 #num categories for quality

#optimizers LRs
tne_init_custom_lr = True
max_tne_general_optimizer_lr = tne_general_optimizer_lr = 0.001
max_quality_layer_lr = quality_layer_lr = 0.001#0.01
max_quality_high_embedding_lr = quality_high_embedding_lr = 0.001#0.01
max_quality_aux_lr = quality_aux_lr = 0.001#0.01
max_quality_aux_logit_lr = quality_aux_logit_lr = 0.001#0.01
max_resonance_layer_lr = resonance_layer_lr = 0.001#0.01
max_resonance_high_embedding_lr = resonance_high_embedding_lr = 0.001#0.01
max_resonance_aux_lr = resonance_aux_lr = 0.001#0.01
max_tension_layer_lr = tension_layer_lr = 0.001#0.01
max_tension_high_embedding_lr = tension_high_embedding_lr = 0.001#0.01
max_tension_aux_lr = tension_aux_lr = 0.001#0.01
max_longevity_layer_lr = longevity_layer_lr = 0.001
max_longevity_high_embedding_lr = longevity_high_embedding_lr = 0.001#0.01
max_longevity_aux_lr = longevity_aux_lr = 0.001#0.01
max_attention_layer_lr = attention_layer_lr = 0.001#0.01
max_combined_layer_lr = combined_layer_lr = 0.001#0.01

tne_tension_layer_dropout = 0.3
tne_resonance_layer_dropout = 0.3
tne_time_layer_dropout = 0.3
tne_combined_layer_dropout = 0.3

#AdamW optimizer decay params
tne_general_optimizer_weight_decay = 0.00001 #flat applied to all layers
quality_layer_weight_decay = 0.00001
quality_high_embedding_weight_decay = 0.00001
quality_aux_weight_decay = 0.00001
quality_aux_logit_weight_decay = 0.00001
resonance_layer_weight_decay = 0.00001
resonance_high_embedding_weight_decay = 0.00001
resonance_aux_weight_decay = 0.00001
tension_layer_weight_decay = 0.000005
tension_high_embedding_weight_decay = 0.000005
tension_aux_weight_decay = 0.000005
longevity_layer_weight_decay = 0.00001
longevity_high_embedding_weight_decay = 0.00001
longevity_aux_weight_decay = 0.00001
combined_layer_weight_decay = 0.0001

clip_aux_layers_grad_max_norm = 0#0.3

#Encoder dimensions
longevity_normd_input_dim = 2
numeric_normd_input_dim = 1

tne_low_level_category_feature_output_dim = 8
tne_low_level_tension_output_dim = 64
tne_low_level_resonance_output_dim = 64
tne_low_level_time_output_dim = 16

tne_high_level_category_feature_output_dim = 64
tne_high_level_continuous_output_dim = 64
tne_high_level_time_output_dim = 64
tne_high_level_all_output_dim = 64

tne_combined_layer_linear_1_output_dim = 512
tne_combined_layer_final_output_dim = 128
#endregion

###############################################################################
# VISUALIZATION
###############################################################################
#region VISUALIZATION: Careful - possible memory issues at per epoch graph
mid_training_operations_epoch_freq = 10 #set mid_training_operations_epoch_freq>target if create_emb_maniform=True
create_emb_maniform = False #if True to manage memory set mid_training_operations_epoch_freq>target # epochs
embed_mp4 = r"C:\ProgramData\miniconda3\envs\entity-rel\Library\bin\ffmpeg.exe"
#endregion

###############################################################################
# MODEL CHECKPOINTS
###############################################################################
#region MODEL CHECKPOINTS
tabular_numeric_encoder_checkpoint_fname = "tabular_numeric_encoder_checkpoint.pth"
#endregion