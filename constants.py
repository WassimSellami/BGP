import os

class Constants:
    TIME_WINDOW = 1
    MA_WINDOW = 20
    SEQUENCE_LENGTH = 24
    PLOT_WINDOW = 30
    SEED = 42
    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    SEQUENCE_LENGTH = 24
    EVALUATION_INTERVAL = 150
    EPOCHS = 100
    FEATURE_NB_A = 'nb_A'         
    FEATURE_NB_W = 'nb_W'
    FEATURE_NB_A_W = 'nb_A_W'
    CHOSEN_COLLECTOR = "rrc12"
    MAX_TRAIN_ROWS = 150000  # Maximum number of rows to use for training and testing