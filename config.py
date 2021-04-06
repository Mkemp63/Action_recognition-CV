import os

# Directory information
ROOT_DIR = os.getcwd()

STANF = os.path.join(ROOT_DIR, 'Data\\Stanford40\\')
STANF_IMG = os.path.join(STANF, 'JPEGImages\\')
STANF_CONV = os.path.join(STANF, 'ImagesConv\\')
STANF_CONV_CROP = os.path.join(STANF, 'ImagesConvCrop\\')

TV = os.path.join(ROOT_DIR, 'Data\\TV-HI\\')
TV_VIDEOS = os.path.join(TV, 'tv_human_interactions_videos')
TV_VIDEOS_SLASH = os.path.join(TV, 'tv_human_interactions_videos\\')

MODELS = os.path.join(ROOT_DIR, "Models\\")
# Data information
Image_size = 112
Use_converted = True

# Divsision
Validate_perc = 0.1

# Training information
Max_epochs = 15
Epochs = 40
Folds = 5
Batch_size = 32
Use_pretrained = True
Test_performance = False
Evaluate_fashion = False
