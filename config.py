# General settings
RANDOM_SEED = 42
DEVICE = "cuda"  # Options: "cuda" or "cpu"

# Model settings
# MODEL_NAME = "FacebookAI/roberta-base"
MODEL_NAME ="google-bert/bert-large-uncased"
MAX_LEN = 512
DROPOUT_RATE = 0.1

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 100
SCHEDULER_STEP_SIZE = 100
SCHEDULER_GAMMA = 0.1

# Evaluation settings
THRESHOLD = 0.10


# Paths
TRAIN_PATH = "data/EN_annotations.csv"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "./saved_models"

# deberta v3 base exp 6
# Exact Match Ratio	micro P	micro R	micro F1	Accuracy for main role
# 0.13190	0.14290	0.13000	0.13610	0.80220

# xlm-roberta-base exp 12
# Exact Match Ratio	micro P	micro R	micro F1	Accuracy for main role
# 0.06590	0.08790	0.08000	0.08380	0.80220

# Bert base uncased exp 13
# Exact Match Ratio	micro P	micro R	micro F1	Accuracy for main role
# 0.12090	0.14740	0.14000	0.14360	0.78020


# Roberta base exp 19
# Exact Match Ratio	micro P	micro R	micro F1	Accuracy for main role
# 0.06590	0.06590	0.06000	0.06280	0.51650

# Bert large uncased exp 21
#Date	Exact Match Ratio	micro P	micro R	micro F1	Accuracy for main role
#December 4 17:30:25	0.17580	0.20880	0.19000	0.19900	0.76920