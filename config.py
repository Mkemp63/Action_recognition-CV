import os

# Directory information
ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, 'models\\')
BASELINE_DIR = os.path.join(MODELS_DIR, 'model_baseline')
DROPOUT_DIR = os.path.join(MODELS_DIR, 'model_dropout')
DECAY_DIR = os.path.join(MODELS_DIR, 'model_lr_decay')
BIGGER_DIR = os.path.join(MODELS_DIR, 'model_bigger_nn')
LESS_DIR = os.path.join(MODELS_DIR, 'model_less')

# Data information
Image_size = 28

# Divsision
Validate_perc = 0.2

# Training information
Max_epochs = 15
Epochs = 15
Folds = 5
Batch_size = 32
Use_pretrained = True
Test_performance = False
Evaluate_fashion = False
