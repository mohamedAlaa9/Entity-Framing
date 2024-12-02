# General settings
RANDOM_SEED = 42
DEVICE = "cuda"  # Options: "cuda" or "cpu"

# Model settings
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
DROPOUT_RATE = 0.3

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 100
SCHEDULER_STEP_SIZE = 100
SCHEDULER_GAMMA = 0.1

# Evaluation settings
THRESHOLD = 0.282

# Paths
TRAIN_PATH = "data/EN_annotations.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "./saved_model"

