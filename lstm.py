import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import pickle
import warnings
warnings.filterwarnings("ignore")
<<<<<<< HEAD:lstm.py

SEED = 42
BATCH_SIZE = 64
BUFFER_SIZE = 100
WINDOW_LENGTH = 24
EVALUATION_INTERVAL = 150
EPOCHS = 100
MODEL_DIR = 'model'
SCALER_DIR = 'scaler'

tf.random.set_seed(SEED)
np.random.seed(SEED)
=======
from constants import (
    FEATURE_NB_A, FEATURE_NB_A_MA, FEATURE_NB_A_W, FEATURE_NB_W, FEATURE_NB_W_MA, SEED, BATCH_SIZE, BUFFER_SIZE,
    EVALUATION_INTERVAL, EPOCHS, MODEL_DIR, SCALER_DIR,
    MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH, TRAINING_DATA_FILE,
    TRAINING_OUTPUT_FILE, TEST_OUTPUT_FILE
)
>>>>>>> 9f4dc35409a68d6ec35588b1d645fac6a800e7f4:prof/lstm.py

plt.style.use('default')
plt.rcParams["figure.figsize"] = (9, 8)

tf.random.set_seed(SEED)
np.random.seed(SEED)

def create_time_features(df, target=None):
    df_1 = pd.DataFrame(df, columns=[FEATURE_NB_A, FEATURE_NB_W, FEATURE_NB_A_W, FEATURE_NB_A_MA, FEATURE_NB_W_MA])
    X = df_1
    
    if target:
        if isinstance(target, list):
            y = df[target].copy()
            return X, y
        else:
            y = df[target].copy()
            return X, y
    return X

def window_data(X, Y, window=SEQUENCE_LENGTH):
    x = []
    y = []
    for i in range(window-1, len(X)):
        x.append(X[i-window+1:i+1])
        if i+1 < len(Y):
            y.append(Y[i+1])
        else:
            y.append(Y[i])
    return np.array(x), np.array(y)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    df = pd.read_csv(TRAINING_DATA_FILE, sep=',', header=0, low_memory=False, 
                    infer_datetime_format=True, parse_dates=True)

    train_size = int(len(df) * 0.8)
    
    df_training = df[1:train_size]
    df_test = df[train_size:]
    print(f"{len(df_training)} days of training data\n{len(df_test)} days of testing data")
<<<<<<< HEAD:lstm.py
    X_train_df, y_train = create_time_features(df_training, target=['nb_A', 'nb_W'])
    X_test_df, y_test = create_time_features(df_test, target=['nb_A', 'nb_W'])
=======

    df_training.to_csv(TRAINING_OUTPUT_FILE)
    df_test.to_csv(TEST_OUTPUT_FILE)

    X_train_df, y_train = create_time_features(df_training, target=[FEATURE_NB_A, FEATURE_NB_W])
    X_test_df, y_test = create_time_features(df_test, target=[FEATURE_NB_A, FEATURE_NB_W])
>>>>>>> 9f4dc35409a68d6ec35588b1d645fac6a800e7f4:prof/lstm.py

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    if not os.path.exists(SCALER_DIR):
        os.makedirs(SCALER_DIR)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {SCALER_PATH}")

    X_w = np.concatenate((X_train, X_test))
    y_w = np.concatenate((y_train, y_test))
    X_w, y_w = window_data(X_w, y_w, window=SEQUENCE_LENGTH)
    
    X_train_w = X_w[:-len(X_test)]
    y_train_w = y_w[:-len(X_test)]
    X_test_w = X_w[-len(X_test):]
    y_test_w = y_w[-len(X_test):]

    train_data = tf.data.Dataset.from_tensor_slices((X_train_w, y_train_w))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((X_test_w, y_test_w))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=X_train_w.shape[-2:], dropout=0.0),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(2)
    ])
    simple_lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    model_history = simple_lstm_model.fit(
        train_data, 
        epochs=EPOCHS,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data=val_data, 
        validation_steps=10
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    simple_lstm_model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    yhat = simple_lstm_model.predict(X_test_w)
    nb_A_pred = yhat[:, 0]
    nb_W_pred = yhat[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 16))
    
    ax1.set_xlabel('Time steps', fontsize=23, fontweight="bold")
    ax1.set_ylabel('Number of Announcements', fontsize=27, fontweight="bold")
    ax1.set_yscale("log")
    ax1.plot(df_test[FEATURE_NB_A].values[4708:4908], label='Original', linewidth=4.0, color='black')
    ax1.plot(nb_A_pred[4708:4908], color='#FF1493', label='LSTM', linewidth=4.0)
    ax1.tick_params(labelsize=15)
    ax1.legend(fontsize=28, loc='upper left')
    ax1.set_title('nb_A Predictions', fontsize=25, fontweight="bold")

    ax2.set_xlabel('Time steps', fontsize=23, fontweight="bold")
    ax2.set_ylabel('Number of Withdrawals', fontsize=27, fontweight="bold")
    ax2.set_yscale("log")
    ax2.plot(df_test[FEATURE_NB_W].values[4708:4908], label='Original', linewidth=4.0, color='black')
    ax2.plot(nb_W_pred[4708:4908], color='#FF1493', label='LSTM', linewidth=4.0)
    ax2.tick_params(labelsize=15)
    ax2.legend(fontsize=28, loc='upper left')
    ax2.set_title('nb_W Predictions', fontsize=25, fontweight="bold")

    plt.tight_layout()
    plt.savefig('prediction_results/train.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Metrics for nb_A:")
    print("RMSE : ", np.sqrt(mean_squared_error(df_test[FEATURE_NB_A], nb_A_pred)), end=",     ")
    print("MAE: ", mean_absolute_error(df_test[FEATURE_NB_A], nb_A_pred), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(df_test[FEATURE_NB_A], nb_A_pred), end=",     ")
    print("r2 : ", r2_score(df_test[FEATURE_NB_A], nb_A_pred))

    print("\nMetrics for nb_W:")
    print("RMSE : ", np.sqrt(mean_squared_error(df_test[FEATURE_NB_W], nb_W_pred)), end=",     ")
    print("MAE: ", mean_absolute_error(df_test[FEATURE_NB_W], nb_W_pred), end=",     ")
    print("MAPE : ", mean_absolute_percentage_error(df_test[FEATURE_NB_W], nb_W_pred), end=",     ")
    print("r2 : ", r2_score(df_test[FEATURE_NB_W], nb_W_pred))

if __name__ == "__main__":
    main()