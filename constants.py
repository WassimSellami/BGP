import os

# Time windows
TIME_WINDOW = 5
MA_WINDOW = 10
SEQUENCE_LENGTH = 24
PLOT_WINDOW = 100

# Model parameters
SEED = 42
BATCH_SIZE = 64
BUFFER_SIZE = 100
EVALUATION_INTERVAL = 150
EPOCHS = 100

# Feature names
FEATURE_NB_A = 'nb_A'         
FEATURE_NB_W = 'nb_W'
FEATURE_NB_A_W = 'nb_A_W'
FEATURE_NB_A_MA = 'nb_A_ma'
FEATURE_NB_W_MA = 'nb_W_ma'

# Collector settings
CHOSEN_COLLECTOR = "rrc12"

# File paths
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join('prof', 'model')
SCALER_DIR = os.path.join('prof', 'scaler')
TEST_DATA_DIR = "test_data"
TRAIN_DATA_DIR = "train_data"

# Training data files
TRAINING_DATA_FILE = os.path.join(TRAIN_DATA_DIR, 'rrc12-ma-5-g3.csv')
TRAINING_OUTPUT_FILE = 'training.csv'
TEST_OUTPUT_FILE = 'test.csv'

# File names
REAL_TIME_FEATURES_FILENAME = os.path.join(OUTPUT_DIR, "real_time_collector_with_predictions.csv")
REAL_TIME_COLLECTOR_FILENAME = os.path.join(OUTPUT_DIR, "real_time_collector.csv")
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.h5')
SCALER_PATH = os.path.join(SCALER_DIR, 'scaler.pkl')

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True) 